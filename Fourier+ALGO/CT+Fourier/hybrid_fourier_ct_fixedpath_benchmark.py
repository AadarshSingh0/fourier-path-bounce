#!/usr/bin/env python3
"""
hybrid_fourier_ct_fixedpath_benchmark.py

Hybrid Fourier-path + CosmoTransitions fixed-path benchmark.

Purpose
-------
This script tests the following workflow:

    1. Define an OptiBounce-like nested multi-field potential.
    2. Find false and true vacua.
    3. Optimize an endpoint-safe Fourier path between the vacua.
    4. Use CosmoTransitions in two ways:

       (A) Full CT path deformation, for dimensions where it is available.
           This is the standard multi-field CosmoTransitions calculation.

       (B) Hybrid fixed-path evaluation.
           The Fourier-optimized path is held fixed. CosmoTransitions' 1D
           bounce solver is then used on the effective potential V(s) along
           that path.

For N_phi <= 10, where full CT path deformation is usually available, the
script compares:

    full CT action/profile  vs  Fourier-fixed-path + CT-1D action/profile.

For larger dimensions, e.g. N_phi = 15, 20, the script runs only the hybrid
fixed-path evaluation. This is not an independent CT multi-field bounce. It is
an action/profile evaluation along the Fourier-optimized path.

Main outputs
------------
outdir/
    hybrid_summary.csv
    action_comparison.png
    relative_difference.png
    profile_compare_N*.png
    hybrid_profile_N*.png
    paths_N*.npz

Important interpretation
------------------------
The hybrid result is a fixed-path bounce action along the Fourier path. It is
not a fully path-deformed CosmoTransitions bounce. The validation test is to
compare the hybrid fixed-path result against full CT for N_phi <= 10.

Dependencies
------------
pip install numpy scipy pandas matplotlib jax jaxlib cosmoTransitions

Example
-------
python3 hybrid_fourier_ct_fixedpath_benchmark.py \
    --nfields-ct 2,3,4,5,6,7,8,9,10 \
    --nfields-hybrid-extra 15,20 \
    --modes 1,2,3,5,8 \
    --outdir hybrid_ct_results
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

try:
    from cosmoTransitions import pathDeformation as ct_path
    from cosmoTransitions.tunneling1D import SingleFieldInstanton
    HAVE_CT = True
except Exception:
    HAVE_CT = False


# =============================================================================
# Plot style
# =============================================================================

def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.facecolor": "#fbfbfb",
        "figure.facecolor": "white",
        "axes.edgecolor": "#2b2b2b",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.facecolor": "white",
        "legend.edgecolor": "#d0d0d0",
        "grid.color": "#b0b0b0",
        "grid.alpha": 0.28,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.3,
        "lines.markersize": 7,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="both")
    ax.tick_params(direction="in", top=True, right=True, length=4)
    for spine in ax.spines.values():
        spine.set_color("#2b2b2b")


# =============================================================================
# Utilities
# =============================================================================

def parse_int_list(s: str) -> list[int]:
    if str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def safe_float(x: Any) -> float | None:
    try:
        y = float(x)
        if np.isfinite(y):
            return y
    except Exception:
        pass
    return None


def omega_d(d: float) -> float:
    return 2.0 * np.pi ** (0.5 * d) / math.gamma(0.5 * d)


def tunneling_prefactor(d: float) -> float:
    return ((d - 1.0) ** (d - 1.0)) * omega_d(d) / d


def cumulative_arc_length(path: np.ndarray) -> np.ndarray:
    dphi = np.diff(path, axis=0)
    ds = np.linalg.norm(dphi, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s


def remove_duplicate_s(path: np.ndarray, s: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(s), dtype=bool)
    keep[1:] = np.diff(s) > eps
    return path[keep], s[keep]


# =============================================================================
# OptiBounce-like potential
# =============================================================================

class OptiBounceLikePotential:
    """Nested OptiBounce-like polynomial potential.

    V(phi) = [sum_i c_i (phi_i - 1)^2 - delta] * [sum_i phi_i^2]

    The origin is an exact stationary false vacuum for sum(c_i) > delta.
    The true vacuum is found numerically.
    """

    def __init__(self, nfields: int, coeffs: np.ndarray, delta: float = 0.065):
        self.nfields = int(nfields)
        self.c = np.asarray(coeffs[: self.nfields], dtype=float)
        self.delta = float(delta)
        if len(self.c) != self.nfields:
            raise ValueError("Not enough coefficients supplied for this nfields.")
        self.phi_false = np.zeros(self.nfields, dtype=float)
        self.phi_true: np.ndarray | None = None

    def V_np(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        A = np.sum(self.c * (phi - 1.0) ** 2, axis=-1) - self.delta
        R2 = np.sum(phi ** 2, axis=-1)
        return A * R2

    def dV_np(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        single = False
        if phi.ndim == 1:
            phi = phi[None, :]
            single = True
        A = np.sum(self.c[None, :] * (phi - 1.0) ** 2, axis=1) - self.delta
        R2 = np.sum(phi ** 2, axis=1)
        grad = 2.0 * self.c[None, :] * (phi - 1.0) * R2[:, None] + 2.0 * phi * A[:, None]
        return grad[0] if single else grad

    def make_V_jax(self):
        if not HAVE_JAX:
            raise RuntimeError("JAX is required for Fourier path optimization.")
        c = jnp.array(self.c)
        delta = self.delta

        @jax.jit
        def V_jax(phi):
            A = jnp.sum(c * (phi - 1.0) ** 2, axis=-1) - delta
            R2 = jnp.sum(phi ** 2, axis=-1)
            return A * R2

        return V_jax

    def find_true_vacuum(self, seed: int = 123, n_random: int = 40) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed + 101 * self.nfields)
        guesses: list[np.ndarray] = []
        guesses.append(np.ones(self.nfields))
        guesses.append(1.2 * np.ones(self.nfields))
        guesses.append(0.7 * np.ones(self.nfields))
        guesses.append(np.full(self.nfields, 0.5))
        guesses.append(np.linspace(0.2, 1.3, self.nfields))

        for _ in range(n_random):
            guesses.append(rng.uniform(-0.4, 1.8, size=self.nfields))

        minima: list[tuple[np.ndarray, float, float]] = []
        for g in guesses:
            try:
                res = minimize(
                    lambda x: float(self.V_np(x)),
                    g,
                    jac=lambda x: self.dV_np(x),
                    method="BFGS",
                    options={"gtol": 1e-10, "maxiter": 5000},
                )
            except Exception:
                continue
            if not np.all(np.isfinite(res.x)):
                continue
            x = np.asarray(res.x, dtype=float)
            val = float(self.V_np(x))
            gnorm = float(np.linalg.norm(self.dV_np(x)))
            if gnorm < 1e-5:
                if not any(np.linalg.norm(x - y[0]) < 1e-7 for y in minima):
                    minima.append((x, val, gnorm))

        if not minima:
            raise RuntimeError(f"Could not find true vacuum for N={self.nfields}.")

        minima.sort(key=lambda tup: tup[1])
        self.phi_true = minima[0][0]
        return minima[0][0], minima[0][1]


def make_nested_coefficients(max_n: int, seed: int = 123, first_two_optibounce: bool = True) -> np.ndarray:
    """Make deterministic nested coefficients for the OptiBounce-like potential."""
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(0.10, 0.80, size=max_n)
    if first_two_optibounce and max_n >= 2:
        coeffs[0] = 0.684373
        coeffs[1] = 0.181928
    return coeffs


# =============================================================================
# Fourier path optimization using fixed-Vt reduced action
# =============================================================================

@dataclass
class FourierPathResult:
    nfields: int
    nmodes: int
    action_proxy: float
    opt_time_sec: float
    success: bool
    message: str
    path_false_to_true: np.ndarray
    path_true_to_false: np.ndarray


def optimize_fourier_path(
    pot: OptiBounceLikePotential,
    nmodes: int,
    n_grid: int = 260,
    maxiter: int = 1000,
    coeff_bound_scale: float = 1.0,
) -> FourierPathResult:
    """Optimize endpoint-safe Fourier path using fixed-Vt reduced action.

    The path is parameterized from false vacuum at t=0 to true vacuum at t=1:

        phi(t) = (1-t) phi_F + t phi_T + sum_n a_n sin(n*pi*t)

    The returned path_true_to_false is the reverse path, suitable for CT 1D:
    s=0 at true vacuum and s=s_end at false vacuum.
    """
    if not HAVE_JAX:
        raise RuntimeError("JAX is not installed. Install jax and jaxlib.")
    if pot.phi_true is None:
        raise RuntimeError("Call pot.find_true_vacuum() before optimizing the path.")

    V_jax = pot.make_V_jax()
    n = pot.nfields
    phi_false = jnp.array(pot.phi_false)
    phi_true = jnp.array(pot.phi_true)
    V_false = V_jax(phi_false)
    V_true = V_jax(phi_true)

    t = jnp.linspace(0.0, 1.0, n_grid)
    ns = jnp.arange(1, nmodes + 1)
    basis = jnp.sin(jnp.pi * t[:, None] * ns[None, :])
    straight = (1.0 - t)[:, None] * phi_false + t[:, None] * phi_true

    # Smooth monotonic tunneling potential interpolation from false to true.
    Vt = t * t * (3.0 - 2.0 * t) * (V_true - V_false)

    @jax.jit
    def action_objective(flat_coeffs):
        coeffs = flat_coeffs.reshape((nmodes, n))
        path = straight + basis @ coeffs
        Vpath = V_jax(path) - V_false

        dphi = path[1:] - path[:-1]
        ds = jnp.sqrt(jnp.sum(dphi * dphi, axis=1) + 1e-30)
        dVt = Vt[1:] - Vt[:-1]
        minus_dVt = -dVt

        # Midpoint average in the form used in the discrete d=4 action:
        # [Vbar - Vtbar]^2 * ds^4 / (-Delta Vt)^3
        Vdiff_mid = 0.5 * ((Vpath[:-1] - Vt[:-1]) + (Vpath[1:] - Vt[1:]))

        good = (Vdiff_mid > 0.0) & (minus_dVt > 0.0)
        bad_v = jnp.minimum(Vdiff_mid, 0.0)
        bad_d = jnp.minimum(minus_dVt, 0.0)
        penalty = 1e12 * (jnp.sum(bad_v * bad_v) + jnp.sum(bad_d * bad_d))

        terms = (Vdiff_mid ** 2) * (ds ** 4) / (minus_dVt ** 3)
        terms = jnp.where(good, terms, 0.0)
        S = 54.0 * jnp.pi ** 2 * jnp.sum(terms)

        # Gentle smoothness regularizer only to avoid pathological optimizer excursions.
        second = path[2:] - 2.0 * path[1:-1] + path[:-2]
        smooth = 1e-6 * jnp.sum(second * second)

        box = 1e-6 * jnp.sum(jnp.maximum(jnp.abs(path) - 5.0, 0.0) ** 2)
        return S + penalty + smooth + box

    val_and_grad = jax.jit(jax.value_and_grad(action_objective))

    def scipy_fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = val_and_grad(jnp.array(x))
        return float(val), np.asarray(grad, dtype=float)

    x0 = np.zeros((nmodes, n), dtype=float).ravel()
    _ = scipy_fun(x0)  # JIT warm-up not counted? We count full user-facing time below.

    dist = float(np.linalg.norm(np.asarray(pot.phi_true) - pot.phi_false))
    bound = coeff_bound_scale * max(dist, 1.0)
    bounds = [(-bound, bound)] * x0.size

    t0 = time.perf_counter()
    res = minimize(
        fun=lambda x: scipy_fun(x)[0],
        x0=x0,
        jac=lambda x: scipy_fun(x)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
    )
    t1 = time.perf_counter()

    coeffs = np.asarray(res.x, dtype=float).reshape((nmodes, n))
    tt = np.linspace(0.0, 1.0, n_grid)
    bb = np.array([np.sin((k + 1) * np.pi * tt) for k in range(nmodes)]).T
    straight_np = (1.0 - tt)[:, None] * pot.phi_false[None, :] + tt[:, None] * pot.phi_true[None, :]
    path_false_to_true = straight_np + bb @ coeffs
    path_false_to_true[0] = pot.phi_false
    path_false_to_true[-1] = pot.phi_true
    path_true_to_false = path_false_to_true[::-1].copy()

    return FourierPathResult(
        nfields=pot.nfields,
        nmodes=nmodes,
        action_proxy=float(res.fun),
        opt_time_sec=t1 - t0,
        success=bool(res.success),
        message=str(res.message),
        path_false_to_true=path_false_to_true,
        path_true_to_false=path_true_to_false,
    )


def scan_fourier_modes(
    pot: OptiBounceLikePotential,
    modes: list[int],
    n_grid: int,
    maxiter: int,
) -> FourierPathResult:
    best: FourierPathResult | None = None
    for m in modes:
        print(f"    Fourier optimizing mode m={m} ...", flush=True)
        r = optimize_fourier_path(pot, m, n_grid=n_grid, maxiter=maxiter)
        print(
            f"      m={m}: proxy={r.action_proxy:.8g}, success={r.success}, time={r.opt_time_sec:.3f}s",
            flush=True,
        )
        if best is None or r.action_proxy < best.action_proxy:
            best = r
    assert best is not None
    print(f"    selected mode m={best.nmodes}, proxy={best.action_proxy:.8g}", flush=True)
    return best


# =============================================================================
# CosmoTransitions wrappers
# =============================================================================

def extract_ct_stdout_info(text: str) -> tuple[float | None, int | None, int | None]:
    matches = re.findall(
        r"Path deformation converged\.\s*(\d+)\s*steps\.\s*fRatio\s*=\s*([0-9.eE+-]+)",
        text,
    )
    if not matches:
        return None, None, None
    steps = [int(m[0]) for m in matches]
    frs = [float(m[1]) for m in matches]
    return frs[-1], int(np.sum(steps)), len(matches)


def recursive_find_action(obj: Any, max_depth: int = 6) -> float | None:
    seen: set[int] = set()
    action_names = {"action", "S", "SE", "S_E", "action1D", "action2D", "tunneling_action"}

    def rec(x: Any, depth: int) -> float | None:
        if x is None or depth > max_depth:
            return None
        oid = id(x)
        if oid in seen:
            return None
        seen.add(oid)

        if isinstance(x, dict):
            for k, v in x.items():
                ks = str(k)
                if ks in action_names or "action" in ks.lower():
                    if callable(v):
                        try:
                            val = safe_float(v())
                            if val is not None:
                                return val
                        except Exception:
                            pass
                    val = safe_float(v)
                    if val is not None:
                        return val
            for v in x.values():
                val = rec(v, depth + 1)
                if val is not None:
                    return val
            return None

        for name in ["action", "S", "SE", "S_E", "findAction", "action1D", "action2D"]:
            if hasattr(x, name):
                try:
                    attr = getattr(x, name)
                    val = safe_float(attr() if callable(attr) else attr)
                    if val is not None:
                        return val
                except Exception:
                    pass

        if hasattr(x, "__dict__"):
            return rec(vars(x), depth + 1)
        return None

    return rec(obj, 0)


def recursive_find_profile(obj: Any, max_depth: int = 6) -> Any | None:
    """Try to find a CT profile object or array in a nested output."""
    seen: set[int] = set()

    def looks_like_profile(x: Any) -> bool:
        return (hasattr(x, "R") and (hasattr(x, "Phi") or hasattr(x, "phi")))

    def rec(x: Any, depth: int) -> Any | None:
        if x is None or depth > max_depth:
            return None
        oid = id(x)
        if oid in seen:
            return None
        seen.add(oid)

        if looks_like_profile(x):
            return x

        if isinstance(x, dict):
            for key in ["profile1D", "profile", "prof", "bounce", "instanton"]:
                if key in x:
                    y = rec(x[key], depth + 1)
                    if y is not None:
                        return y
            for v in x.values():
                y = rec(v, depth + 1)
                if y is not None:
                    return y
            return None

        if hasattr(x, "__dict__"):
            return rec(vars(x), depth + 1)
        return None

    return rec(obj, 0)


def profile_to_arrays(profile: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if profile is None:
        return None, None
    R = None
    Phi = None
    for name in ["R", "r", "radius"]:
        if hasattr(profile, name):
            R = np.asarray(getattr(profile, name), dtype=float)
            break
    for name in ["Phi", "phi", "field"]:
        if hasattr(profile, name):
            Phi = np.asarray(getattr(profile, name), dtype=float)
            break
    return R, Phi


def run_ct_full_deformation(pot: OptiBounceLikePotential, path_true_to_false: np.ndarray, maxiter: int = 40) -> dict[str, Any]:
    if not HAVE_CT:
        return {"success": False, "error": "cosmoTransitions is not installed"}

    t0 = time.perf_counter()
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = ct_path.fullTunneling(
                path_true_to_false,
                pot.V_np,
                pot.dV_np,
                maxiter=maxiter,
                verbose=False,
            )
        log = buf.getvalue()
        t1 = time.perf_counter()
        fRatio, steps_total, n_deform = extract_ct_stdout_info(log)
        action = recursive_find_action(out)
        profile = recursive_find_profile(out)
        R, Phi = profile_to_arrays(profile)
        return {
            "success": True,
            "action": action,
            "fRatio": fRatio,
            "steps_total": steps_total,
            "n_deformations": n_deform,
            "time_sec": t1 - t0,
            "stdout": log,
            "raw_output_type": str(type(out)),
            "profile_R": R,
            "profile_Phi": Phi,
            "error": None,
        }
    except Exception as exc:
        t1 = time.perf_counter()
        return {
            "success": False,
            "action": None,
            "fRatio": None,
            "steps_total": None,
            "n_deformations": None,
            "time_sec": t1 - t0,
            "stdout": "",
            "raw_output_type": None,
            "profile_R": None,
            "profile_Phi": None,
            "error": str(exc) + "\n" + traceback.format_exc(limit=2),
        }


# =============================================================================
# Fixed-path CT 1D evaluator
# =============================================================================

@dataclass
class FixedPathEvaluator:
    path_true_to_false: np.ndarray
    s: np.ndarray
    V_of_s: Callable[[np.ndarray], np.ndarray]
    dVds_of_s: Callable[[np.ndarray], np.ndarray]
    phi_of_s: list[PchipInterpolator]
    s_true: float
    s_false: float


def build_fixed_path_evaluator(pot: OptiBounceLikePotential, path_true_to_false: np.ndarray) -> FixedPathEvaluator:
    path = np.asarray(path_true_to_false, dtype=float)
    s = cumulative_arc_length(path)
    path, s = remove_duplicate_s(path, s)

    if len(s) < 5 or s[-1] <= 0:
        raise ValueError("Bad Fourier path: arc length is zero or too few points.")

    Vvals = np.asarray(pot.V_np(path), dtype=float)
    V_interp = PchipInterpolator(s, Vvals, extrapolate=True)
    dV_interp = V_interp.derivative()
    phi_interp = [PchipInterpolator(s, path[:, i], extrapolate=True) for i in range(path.shape[1])]

    def V_of_s(x):
        xx = np.asarray(x, dtype=float)
        return V_interp(np.clip(xx, s[0], s[-1]))

    def dVds_of_s(x):
        xx = np.asarray(x, dtype=float)
        return dV_interp(np.clip(xx, s[0], s[-1]))

    return FixedPathEvaluator(
        path_true_to_false=path,
        s=s,
        V_of_s=V_of_s,
        dVds_of_s=dVds_of_s,
        phi_of_s=phi_interp,
        s_true=float(s[0]),
        s_false=float(s[-1]),
    )


def lift_s_profile_to_fields(eval1d: FixedPathEvaluator, s_profile: np.ndarray) -> np.ndarray:
    s_clipped = np.clip(np.asarray(s_profile, dtype=float), eval1d.s_true, eval1d.s_false)
    cols = [interp(s_clipped) for interp in eval1d.phi_of_s]
    return np.vstack(cols).T


def compute_perp_force_ratio(pot: OptiBounceLikePotential, eval1d: FixedPathEvaluator) -> float:
    """Diagnostic only: ratio of perpendicular gradient to total gradient along fixed path.

    This is not the same as CosmoTransitions' internal fRatio, but it is useful for
    diagnosing whether the fixed path is close to a valley of the potential.
    """
    path = eval1d.path_true_to_false
    grad = pot.dV_np(path)
    dpath = np.gradient(path, eval1d.s, axis=0, edge_order=1)
    tnorm = np.linalg.norm(dpath, axis=1)
    tangent = dpath / np.maximum(tnorm[:, None], 1e-30)
    gpar = np.sum(grad * tangent, axis=1)[:, None] * tangent
    gperp = grad - gpar
    num = np.sqrt(np.mean(np.sum(gperp * gperp, axis=1)))
    den = np.sqrt(np.mean(np.sum(grad * grad, axis=1))) + 1e-30
    return float(num / den)


def run_ct_1d_fixed_path(pot: OptiBounceLikePotential, path_true_to_false: np.ndarray) -> dict[str, Any]:
    if not HAVE_CT:
        return {"success": False, "error": "cosmoTransitions is not installed"}

    t0 = time.perf_counter()
    try:
        eval1d = build_fixed_path_evaluator(pot, path_true_to_false)

        # True vacuum at s=0, false vacuum at s=s_end.
        instanton = SingleFieldInstanton(
            eval1d.s_true,
            eval1d.s_false,
            eval1d.V_of_s,
            dV=eval1d.dVds_of_s,
        )
        profile = instanton.findProfile()
        action = float(instanton.findAction(profile))
        t1 = time.perf_counter()

        R, s_profile = profile_to_arrays(profile)
        if s_profile is not None and s_profile.ndim > 1:
            s_profile = np.ravel(s_profile)
        Phi_profile = None
        if s_profile is not None:
            Phi_profile = lift_s_profile_to_fields(eval1d, s_profile)

        return {
            "success": True,
            "action": action,
            "time_sec": t1 - t0,
            "profile_R": R,
            "profile_s": s_profile,
            "profile_Phi": Phi_profile,
            "path_s": eval1d.s,
            "path_Phi": eval1d.path_true_to_false,
            "perp_force_ratio": compute_perp_force_ratio(pot, eval1d),
            "error": None,
        }
    except Exception as exc:
        t1 = time.perf_counter()
        return {
            "success": False,
            "action": None,
            "time_sec": t1 - t0,
            "profile_R": None,
            "profile_s": None,
            "profile_Phi": None,
            "path_s": None,
            "path_Phi": None,
            "perp_force_ratio": None,
            "error": str(exc) + "\n" + traceback.format_exc(limit=2),
        }


# =============================================================================
# Plotting outputs
# =============================================================================

def plot_action_comparison(summary: pd.DataFrame, outdir: Path) -> None:
    apply_plot_style()
    df = summary.copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    has_ct = df["ct_full_success"] == True
    has_h = df["hybrid_success"] == True

    if has_ct.any():
        ax.plot(
            df.loc[has_ct, "nfields"],
            df.loc[has_ct, "ct_full_action"],
            marker="s",
            linestyle="--",
            color="#222222",
            label="Full CT path deformation",
        )
    if has_h.any():
        ax.plot(
            df.loc[has_h, "nfields"],
            df.loc[has_h, "hybrid_action"],
            marker="o",
            linestyle="-",
            color="#1f4aff",
            label="Fourier path + CT 1D",
        )

    ax.set_xlabel(r"number of fields $N_\phi$")
    ax.set_ylabel(r"action $S_E$")
    ax.set_title("Full CT versus Fourier-fixed-path CT evaluation")
    ax.set_xticks(sorted(df["nfields"].unique()))
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "action_comparison.png")
    fig.savefig(outdir / "action_comparison.pdf")
    plt.close(fig)


def plot_relative_difference(summary: pd.DataFrame, outdir: Path) -> None:
    apply_plot_style()
    df = summary.copy()
    ok = (df["ct_full_success"] == True) & (df["hybrid_success"] == True)
    df = df[ok].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    rel = 100.0 * np.abs(df["hybrid_action"] - df["ct_full_action"]) / np.abs(df["ct_full_action"])
    ax.plot(df["nfields"], rel, marker="o", color="#c7361d")
    ax.set_xlabel(r"number of fields $N_\phi$")
    ax.set_ylabel(r"$|S_{\rm hybrid}-S_{\rm CT}|/S_{\rm CT}$ [\%]")
    ax.set_title("Fixed-path hybrid action difference where full CT is available")
    ax.set_xticks(sorted(df["nfields"].unique()))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(outdir / "relative_difference.png")
    fig.savefig(outdir / "relative_difference.pdf")
    plt.close(fig)


def _profile_scalar_from_multifield(Phi: np.ndarray | None) -> np.ndarray | None:
    if Phi is None:
        return None
    arr = np.asarray(Phi, dtype=float)
    if arr.ndim == 1:
        return arr
    return np.linalg.norm(arr - arr[-1][None, :], axis=1)


def plot_profile_compare(nfields: int, ct: dict[str, Any], hybrid: dict[str, Any], outdir: Path) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    plotted = False

    R_ct = ct.get("profile_R")
    Phi_ct = ct.get("profile_Phi")
    y_ct = _profile_scalar_from_multifield(Phi_ct)
    if R_ct is not None and y_ct is not None:
        ax.plot(R_ct, y_ct, linestyle="--", color="#222222", label="Full CT")
        plotted = True

    R_h = hybrid.get("profile_R")
    Phi_h = hybrid.get("profile_Phi")
    y_h = _profile_scalar_from_multifield(Phi_h)
    if R_h is not None and y_h is not None:
        ax.plot(R_h, y_h, linestyle="-", color="#1f4aff", label="Fourier path + CT 1D")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(r"Euclidean radius $r$")
    ax.set_ylabel(r"projected field distance from false vacuum")
    ax.set_title(fr"Bounce profile comparison, $N_\phi={nfields}$")
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"profile_compare_N{nfields}.png")
    fig.savefig(outdir / f"profile_compare_N{nfields}.pdf")
    plt.close(fig)


def plot_hybrid_profile(nfields: int, hybrid: dict[str, Any], outdir: Path) -> None:
    apply_plot_style()
    R_h = hybrid.get("profile_R")
    Phi_h = hybrid.get("profile_Phi")
    y_h = _profile_scalar_from_multifield(Phi_h)
    if R_h is None or y_h is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(R_h, y_h, color="#1f4aff", label="Fourier path + CT 1D")
    ax.set_xlabel(r"Euclidean radius $r$")
    ax.set_ylabel(r"projected field distance from false vacuum")
    ax.set_title(fr"Hybrid fixed-path bounce profile, $N_\phi={nfields}$")
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"hybrid_profile_N{nfields}.png")
    fig.savefig(outdir / f"hybrid_profile_N{nfields}.pdf")
    plt.close(fig)


# =============================================================================
# Main benchmark driver
# =============================================================================

def run_one_dimension(
    N: int,
    coeffs: np.ndarray,
    delta: float,
    seed: int,
    modes: list[int],
    n_grid: int,
    maxiter_fourier: int,
    full_ct: bool,
    ct_maxiter: int,
    outdir: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"N_phi = {N}")
    print("=" * 80)

    pot = OptiBounceLikePotential(N, coeffs=coeffs, delta=delta)
    true, Vtrue = pot.find_true_vacuum(seed=seed)
    false = pot.phi_false
    Vfalse = float(pot.V_np(false))
    print(f"false V = {Vfalse:.12g}, |grad| = {np.linalg.norm(pot.dV_np(false)):.3e}")
    print(f"true  V = {Vtrue:.12g}, |grad| = {np.linalg.norm(pot.dV_np(true)):.3e}")
    print(f"Delta V = {Vfalse - Vtrue:.12g}")

    fourier = scan_fourier_modes(pot, modes=modes, n_grid=n_grid, maxiter=maxiter_fourier)

    # Hybrid fixed-path 1D CT evaluation.
    print("    Running hybrid fixed-path CT 1D evaluation ...", flush=True)
    hybrid = run_ct_1d_fixed_path(pot, fourier.path_true_to_false)
    print(
        f"      hybrid success={hybrid.get('success')}, action={hybrid.get('action')}, "
        f"perp_ratio={hybrid.get('perp_force_ratio')}, time={hybrid.get('time_sec')}",
        flush=True,
    )

    # Full CT path deformation only for chosen N.
    if full_ct:
        print("    Running full CosmoTransitions path deformation ...", flush=True)
        straight_true_to_false = np.linspace(true, false, n_grid)
        ct = run_ct_full_deformation(pot, straight_true_to_false, maxiter=ct_maxiter)
        print(
            f"      full CT success={ct.get('success')}, action={ct.get('action')}, "
            f"fRatio={ct.get('fRatio')}, steps={ct.get('steps_total')}, time={ct.get('time_sec')}",
            flush=True,
        )
    else:
        ct = {
            "success": False,
            "action": None,
            "fRatio": None,
            "steps_total": None,
            "n_deformations": None,
            "time_sec": None,
            "profile_R": None,
            "profile_Phi": None,
            "error": "full CT not requested for this N",
        }

    # Save paths and profiles for this N.
    np.savez_compressed(
        outdir / f"paths_N{N}.npz",
        phi_false=false,
        phi_true=true,
        fourier_path_false_to_true=fourier.path_false_to_true,
        fourier_path_true_to_false=fourier.path_true_to_false,
        hybrid_profile_R=hybrid.get("profile_R"),
        hybrid_profile_s=hybrid.get("profile_s"),
        hybrid_profile_Phi=hybrid.get("profile_Phi"),
        ct_profile_R=ct.get("profile_R"),
        ct_profile_Phi=ct.get("profile_Phi"),
    )

    if full_ct:
        plot_profile_compare(N, ct, hybrid, outdir)
    else:
        plot_hybrid_profile(N, hybrid, outdir)

    rel_diff = None
    if ct.get("success") and hybrid.get("success") and ct.get("action") is not None and hybrid.get("action") is not None:
        rel_diff = abs(float(hybrid["action"]) - float(ct["action"])) / abs(float(ct["action"]))

    return {
        "nfields": N,
        "delta": delta,
        "V_false": Vfalse,
        "V_true": Vtrue,
        "deltaV": Vfalse - Vtrue,
        "fourier_nmodes": fourier.nmodes,
        "fourier_proxy_action": fourier.action_proxy,
        "fourier_opt_success": fourier.success,
        "fourier_opt_time_sec": fourier.opt_time_sec,
        "hybrid_success": bool(hybrid.get("success")),
        "hybrid_action": hybrid.get("action"),
        "hybrid_time_sec": hybrid.get("time_sec"),
        "hybrid_perp_force_ratio": hybrid.get("perp_force_ratio"),
        "hybrid_error": hybrid.get("error"),
        "ct_full_requested": bool(full_ct),
        "ct_full_success": bool(ct.get("success")),
        "ct_full_action": ct.get("action"),
        "ct_full_fRatio": ct.get("fRatio"),
        "ct_full_steps": ct.get("steps_total"),
        "ct_full_time_sec": ct.get("time_sec"),
        "ct_full_error": ct.get("error"),
        "rel_action_diff_hybrid_vs_ct": rel_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Fourier-path + CosmoTransitions fixed-path benchmark.")
    parser.add_argument("--nfields-ct", default="2,3,4,5,6,7,8,9,10", help="N values for full CT + hybrid comparison.")
    parser.add_argument("--nfields-hybrid-extra", default="15,20", help="N values for hybrid-only fixed-path runs.")
    parser.add_argument("--modes", default="1,2,3,5,8", help="Fourier mode scan list.")
    parser.add_argument("--delta", type=float, default=0.065, help="OptiBounce-like delta parameter.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for nested random coefficients and minimization starts.")
    parser.add_argument("--n-grid", type=int, default=260, help="Number of points in path grid.")
    parser.add_argument("--maxiter-fourier", type=int, default=1200, help="Max L-BFGS-B iterations for Fourier path optimization.")
    parser.add_argument("--ct-maxiter", type=int, default=40, help="Max path-deformation iterations for full CT.")
    parser.add_argument("--outdir", default="hybrid_ct_results", help="Output directory.")
    parser.add_argument("--no-full-ct", action="store_true", help="Skip full CT even for nfields-ct values.")
    args = parser.parse_args()

    if not HAVE_JAX:
        raise RuntimeError("JAX is not installed. Install jax and jaxlib.")
    if not HAVE_CT:
        raise RuntimeError("cosmoTransitions is not installed. Install cosmoTransitions.")

    n_ct = parse_int_list(args.nfields_ct)
    n_extra = parse_int_list(args.nfields_hybrid_extra)
    modes = parse_int_list(args.modes)
    all_n = sorted(set(n_ct + n_extra))
    max_n = max(all_n)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    coeffs = make_nested_coefficients(max_n, seed=args.seed, first_two_optibounce=True)
    np.savetxt(outdir / "coefficients.txt", coeffs)
    with open(outdir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    rows = []
    for N in all_n:
        full_ct = (N in n_ct) and (not args.no_full_ct)
        row = run_one_dimension(
            N=N,
            coeffs=coeffs,
            delta=args.delta,
            seed=args.seed,
            modes=modes,
            n_grid=args.n_grid,
            maxiter_fourier=args.maxiter_fourier,
            full_ct=full_ct,
            ct_maxiter=args.ct_maxiter,
            outdir=outdir,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(outdir / "hybrid_summary_partial.csv", index=False)

    summary = pd.DataFrame(rows).sort_values("nfields")
    summary.to_csv(outdir / "hybrid_summary.csv", index=False)
    print("\nSaved", outdir / "hybrid_summary.csv")
    print(summary.to_string(index=False))

    plot_action_comparison(summary, outdir)
    plot_relative_difference(summary, outdir)

    print("\nDone.")
    print("Key outputs:")
    print("  ", outdir / "hybrid_summary.csv")
    print("  ", outdir / "action_comparison.png")
    print("  ", outdir / "relative_difference.png")
    print("  ", outdir / "profile_compare_N*.png")
    print("  ", outdir / "hybrid_profile_N*.png")
    print("\nInterpretation reminder:")
    print("  Hybrid = Fourier-optimized fixed path + CosmoTransitions 1D bounce solver.")
    print("  For N > CT range, this is not an independent CT multi-field path-deformation result.")


if __name__ == "__main__":
    main()
