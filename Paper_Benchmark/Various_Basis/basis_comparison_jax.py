#!/usr/bin/env python3
"""
basis_comparison_jax.py

Adaptive JAX-JIT basis comparison for bounce-path minimization.

Default: published OptiBounce benchmark potential in d=3.
It compares endpoint-safe bases and saves coefficients/paths at every mode.

Install:
  python3 -m pip install "jax[cpu]" scipy pandas numpy matplotlib
Run:
  python3 basis_comparison_jax.py --run-mode optibounce_D3
  python3 basis_comparison_jax.py --run-mode optibounce_D4
  python3 basis_comparison_jax.py --run-mode mega_random_D4

Outputs:
  results/<RUN_MODE>/basis_comparison_history.csv
  results/<RUN_MODE>/basis_comparison_summary.csv
  results/<RUN_MODE>/<case>__<basis>/mode_XXX.npz
"""
from __future__ import annotations

import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from math import gamma
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# =============================================================================
# USER SETTINGS
# =============================================================================

# "optibounce_paper" or "mega_random_csv"
POTENTIAL_NAME = "optibounce_paper"
ACTION_DIM = 3                         # d=3 for published OptiBounce table; d=4 for O(4) action
NPHI_VALUES = list(range(3, 21))

BASIS_LIST = [
    "fourier",
    "chebyshev",
    "legendre",
    "bspline_local",
    "gaussian_local",
    "hybrid_fourier_gaussian",
    "hybrid_fourier_bspline",
    "wavelet_db2",
    "wavelet_db4",
    "wavelet_db6",
    "wavelet_sym4",
    "wavelet_coif1",
]

N_BASIS_START = 3
N_BASIS_MAX = 20
N_BASIS_STEP = 1
TOLERANCE = 1.0e-3
PATIENCE = 3

N_GRID = 260
MAXITER = 800
GTOL = 1.0e-7
FTOL = 1.0e-10

USE_ZERO_START = True
USE_PREVIOUS_WARM_START = True
USE_BEST_WARM_START = False
N_RANDOM_STARTS = 0
RANDOM_SCALE = 0.03
BOUND_FACTOR = 0.5

RUN_LABEL = f"{POTENTIAL_NAME}_D{ACTION_DIM}"
OUTDIR = Path("results") / RUN_LABEL
SAVE_EVERY_MODE_NPZ = True
MEGA_RANDOM_CSV = "input_data/mega_random_coefficients.csv"

# =============================================================================
# PUBLISHED OPTIBOUNCE DATA, d=3 reference table
# =============================================================================

OPTIBOUNCE_DATA = {
    3: {"delta": 0.065, "c": [0.684373, 0.181928, 0.295089], "S_FB_D3": 240.403},
    4: {"delta": 0.11, "c": [0.534808, 0.77023, 0.838912, 0.00517238], "S_FB_D3": 230.864},
    5: {"delta": 0.13, "c": [0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "S_FB_D3": 233.716},
    6: {"delta": 0.15, "c": [0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "S_FB_D3": 270.578},
    7: {"delta": 0.2, "c": [0.5233, 0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "S_FB_D3": 250.222},
    8: {"delta": 0.22, "c": [0.2434, 0.5233, 0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.51723], "S_FB_D3": 268.486},
    9: {"delta": 0.29, "c": [0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "S_FB_D3": 204.796},
    10: {"delta": 0.27, "c": [0.12, 0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "S_FB_D3": 261.703},
    11: {"delta": 0.30, "c": [0.23, 0.21, 0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "S_FB_D3": 273.564},
    12: {"delta": 0.32, "c": [0.12, 0.11, 0.12, 0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "S_FB_D3": 249.928},
    13: {"delta": 0.39, "c": [0.54, 0.47, 0.53, 0.28, 0.35, 0.27, 0.42, 0.59, 0.33, 0.16, 0.38, 0.35, 0.17], "S_FB_D3": 293.653},
    14: {"delta": 0.39, "c": [0.39, 0.23, 0.26, 0.40, 0.11, 0.42, 0.41, 0.27, 0.42, 0.54, 0.18, 0.59, 0.13, 0.29], "S_FB_D3": 294.877},
    15: {"delta": 0.42, "c": [0.21, 0.22, 0.22, 0.23, 0.39, 0.55, 0.43, 0.12, 0.16, 0.58, 0.25, 0.50, 0.45, 0.35, 0.45], "S_FB_D3": 313.039},
    16: {"delta": 0.45, "c": [0.42, 0.34, 0.43, 0.22, 0.59, 0.41, 0.58, 0.41, 0.26, 0.45, 0.16, 0.31, 0.39, 0.57, 0.43, 0.10], "S_FB_D3": 366.978},
    17: {"delta": 0.52, "c": [0.24, 0.35, 0.39, 0.56, 0.37, 0.41, 0.52, 0.31, 0.52, 0.22, 0.58, 0.39, 0.39, 0.17, 0.46, 0.30, 0.37], "S_FB_D3": 342.893},
    18: {"delta": 0.47, "c": [0.18, 0.17, 0.30, 0.22, 0.38, 0.48, 0.11, 0.49, 0.43, 0.47, 0.21, 0.29, 0.32, 0.36, 0.30, 0.56, 0.46, 0.42], "S_FB_D3": 391.231},
    19: {"delta": 0.56, "c": [0.40, 0.14, 0.10, 0.43, 0.39, 0.27, 0.33, 0.59, 0.48, 0.36, 0.24, 0.28, 0.51, 0.59, 0.40, 0.39, 0.24, 0.35, 0.20], "S_FB_D3": 339.467},
    20: {"delta": 0.55, "c": [0.42, 0.11, 0.47, 0.13, 0.16, 0.24, 0.58, 0.53, 0.38, 0.44, 0.18, 0.46, 0.47, 0.27, 0.53, 0.24, 0.33, 0.40, 0.32, 0.29], "S_FB_D3": 382.153},
}

# =============================================================================
# CASES AND POTENTIALS
# =============================================================================

@dataclass
class CaseData:
    label: str
    potential_name: str
    nphi: int
    c: Optional[np.ndarray]
    delta: Optional[float]
    false_vac: np.ndarray
    true_vac: np.ndarray
    Vfalse: float
    Vtrue: float
    DeltaV: float
    refs: Dict[str, Any]
    extra: Dict[str, Any]


def optibounce_V_np(phi, c, delta):
    phi = np.asarray(phi, dtype=float)
    A = np.sum(c * (phi - 1.0) ** 2, axis=-1) - delta
    return A * np.sum(phi ** 2, axis=-1)


def optibounce_grad_np(phi, c, delta):
    phi = np.asarray(phi, dtype=float)
    A = np.sum(c * (phi - 1.0) ** 2) - delta
    R2 = np.sum(phi ** 2)
    return 2.0 * c * (phi - 1.0) * R2 + 2.0 * phi * A


def make_optibounce_case(nphi):
    data = OPTIBOUNCE_DATA[nphi]
    c = np.array(data["c"], dtype=float)
    delta = float(data["delta"])
    false = np.zeros(nphi)

    def obj(x): return float(optibounce_V_np(x, c, delta))
    def jac(x): return optibounce_grad_np(x, c, delta)

    res = minimize(obj, np.ones(nphi), jac=jac, method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
    true = res.x.astype(float)
    Vfalse = obj(false)
    Vtrue = obj(true)
    if Vfalse - Vtrue <= 0:
        raise RuntimeError(f"Bad vacuum ordering for nphi={nphi}")
    return CaseData(f"optibounce_n{nphi}_d{ACTION_DIM}", "optibounce_paper", nphi, c, delta, false, true, Vfalse, Vtrue, Vfalse - Vtrue, data, {})


def load_mega_random_data(path):
    import ast
    df = pd.read_csv(path)
    return {int(r["nphi"]): {"delta": float(r["delta"]), "c": ast.literal_eval(r["c_list"])} for _, r in df.iterrows()}


def make_mega_random_case(nphi):
    data = load_mega_random_data(MEGA_RANDOM_CSV)[nphi]
    c = np.array(data["c"], dtype=float)
    delta = float(data["delta"])
    false = np.zeros(nphi)
    def obj(x): return float(optibounce_V_np(x, c, delta))
    def jac(x): return optibounce_grad_np(x, c, delta)
    res = minimize(obj, np.ones(nphi), jac=jac, method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
    true = res.x.astype(float)
    Vfalse = obj(false); Vtrue = obj(true)
    return CaseData(f"mega_random_n{nphi}_d{ACTION_DIM}", "mega_random_csv", nphi, c, delta, false, true, Vfalse, Vtrue, Vfalse - Vtrue, {}, {})


def make_case(nphi):
    if POTENTIAL_NAME == "optibounce_paper": return make_optibounce_case(nphi)
    if POTENTIAL_NAME == "mega_random_csv": return make_mega_random_case(nphi)
    raise ValueError(POTENTIAL_NAME)

# =============================================================================
# BASIS MATRICES
# =============================================================================

def normalize_rows(B):
    B = np.asarray(B, dtype=float)
    for i in range(B.shape[0]):
        m = np.max(np.abs(B[i]))
        if m > 0: B[i] /= m
    return B


def fourier_basis(t, n):
    return normalize_rows(np.array([np.sin((k+1)*np.pi*t) for k in range(n)]))


def chebyshev_basis(t, n):
    x = 2*t - 1
    env = t*(1-t)
    rows = [env * np.cos((k+1)*np.arccos(np.clip(x, -1, 1))) for k in range(n)]
    return normalize_rows(np.array(rows))


def legendre_basis(t, n):
    from numpy.polynomial.legendre import legval
    x = 2*t - 1
    env = t*(1-t)
    rows = []
    for k in range(n):
        coeff = np.zeros(k+2); coeff[k+1] = 1.0
        rows.append(env * legval(x, coeff))
    return normalize_rows(np.array(rows))


def gaussian_basis(t, n):
    env = t*(1-t)
    centers = np.linspace(0.1, 0.9, n)
    width = 0.75/max(n, 2)
    return normalize_rows(np.array([env*np.exp(-0.5*((t-c)/width)**2) for c in centers]))


def bspline_like_basis(t, n):
    try:
        from scipy.interpolate import BSpline
        degree = min(3, max(0, n-1))
        if n <= degree: return gaussian_basis(t, n)
        n_knots = n + degree + 1
        interior_count = n_knots - 2*(degree+1)
        interior = np.linspace(0, 1, interior_count+2)[1:-1] if interior_count > 0 else []
        knots = np.r_[np.zeros(degree+1), interior, np.ones(degree+1)]
        env = t*(1-t)
        rows = []
        for i in range(n):
            coeff = np.zeros(n); coeff[i] = 1
            rows.append(env * np.nan_to_num(BSpline(knots, coeff, degree, extrapolate=False)(t), nan=0.0))
        return normalize_rows(np.array(rows))
    except Exception:
        return gaussian_basis(t, n)


def hybrid_basis(t, n, kind):
    nf = max(1, int(np.ceil(0.6*n)))
    nl = n - nf
    Bf = fourier_basis(t, nf)
    if nl <= 0: return Bf
    Bl = gaussian_basis(t, nl) if kind == "gaussian" else bspline_like_basis(t, nl)
    return normalize_rows(np.vstack([Bf, Bl]))


def wavelet_basis(t, n_basis, wavelet_name="db4"):
    """
    Endpoint-safe sampled wavelet basis.

    We build localized wavelet-like functions using PyWavelets' wavefun output,
    shift/scale them over [0,1], and multiply by t(1-t) so that each basis
    function vanishes at both endpoints.

    This is not an orthonormal discrete wavelet transform basis. It is a
    practical endpoint-safe wavelet-inspired variational basis for path
    deformation.
    """
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "PyWavelets is needed for wavelet bases. Install with: pip install PyWavelets"
        ) from exc

    envelope = t * (1.0 - t)

    wavelet = pywt.Wavelet(wavelet_name)

    # For orthogonal wavelets, wavefun usually returns phi, psi, x.
    # For some families, return length can differ, so handle generally.
    wf = wavelet.wavefun(level=8)

    if len(wf) == 3:
        phi, psi, xw = wf
    elif len(wf) == 5:
        phi_d, psi_d, phi_r, psi_r, xw = wf
        psi = psi_d
    else:
        raise RuntimeError(f"Unexpected wavefun output for {wavelet_name}: len={len(wf)}")

    xw = np.asarray(xw, dtype=float)
    psi = np.asarray(psi, dtype=float)

    # Normalize wavelet support to [0,1].
    xw_min = np.min(xw)
    xw_max = np.max(xw)
    xw_unit = (xw - xw_min) / (xw_max - xw_min)

    rows = []

    # Put n_basis shifted/scaled copies across [0,1].
    centers = np.linspace(0.1, 0.9, n_basis)

    # Width shrinks mildly with n_basis so higher basis counts can capture local structure.
    width = 0.8 / max(n_basis, 2)

    for c in centers:
        # Map t to wavelet's normalized support around center c.
        u = (t - c) / width + 0.5

        vals = np.interp(u, xw_unit, psi, left=0.0, right=0.0)

        rows.append(envelope * vals)

    return normalize_rows(np.array(rows, dtype=float))


def make_basis_matrix(t, name, n):
    if name == "fourier": return fourier_basis(t, n)
    if name == "chebyshev": return chebyshev_basis(t, n)
    if name == "legendre": return legendre_basis(t, n)
    if name == "gaussian_local": return gaussian_basis(t, n)
    if name == "bspline_local": return bspline_like_basis(t, n)
    if name == "hybrid_fourier_gaussian": return hybrid_basis(t, n, "gaussian")
    if name == "hybrid_fourier_bspline": return hybrid_basis(t, n, "bspline")
    if name == "wavelet_db2": return wavelet_basis(t, n, "db2")
    if name == "wavelet_db4": return wavelet_basis(t, n, "db4")
    if name == "wavelet_db6": return wavelet_basis(t, n, "db6")
    if name == "wavelet_sym4": return wavelet_basis(t, n, "sym4")
    if name == "wavelet_coif1": return wavelet_basis(t, n, "coif1")
    raise ValueError(name)

# =============================================================================
# ACTION AND OPTIMIZATION
# =============================================================================

def omega_d(d): return 2*np.pi**(0.5*d)/gamma(0.5*d)
def action_prefactor(d): return ((d-1)**(d-1))*omega_d(d)/d


def V_shift_np(phi, case):
    if case.potential_name in ("optibounce_paper", "mega_random_csv"):
        return optibounce_V_np(phi, case.c, case.delta) - case.Vfalse
    raise ValueError(case.potential_name)


def build_compiled_value_grad(case, B_np, t_np):
    B = jnp.asarray(B_np); t = jnp.asarray(t_np)
    false = jnp.asarray(case.false_vac); true = jnp.asarray(case.true_vac)
    Vfalse = jnp.asarray(case.Vfalse); Vtrue = jnp.asarray(case.Vtrue)
    nphi, nb = case.nphi, B_np.shape[0]
    d = float(ACTION_DIM); pref = jnp.asarray(action_prefactor(d))

    if case.potential_name in ("optibounce_paper", "mega_random_csv"):
        c = jnp.asarray(case.c); delta = jnp.asarray(case.delta)
        def V_shift(phi):
            A = jnp.sum(c*(phi-1.0)**2, axis=-1) - delta
            return A*jnp.sum(phi**2, axis=-1) - Vfalse
    else:
        eps = float(case.extra["eps"]); Aamp = float(case.extra["A"]); my = float(case.extra["my"]); width = float(case.extra["width"])
        def V_shift(phi):
            x = phi[...,0]; y = phi[...,1]
            y0 = Aamp*jnp.tanh(x/width)
            return 0.25*(x*x-1)**2 - eps*x + 0.5*my*my*(y-y0)**2 - Vfalse

    def action(coeffs_flat):
        coeffs = coeffs_flat.reshape((nphi, nb))
        straight = false[None,:] + t[:,None]*(true-false)[None,:]
        path = straight + B.T @ coeffs.T
        Vpath = V_shift(path)
        Vt = Vfalse + t*t*(3-2*t)*(Vtrue - Vfalse)
        dVt = Vt[1:] - Vt[:-1]
        minus_dVt = -dVt
        ds = jnp.linalg.norm(path[1:] - path[:-1], axis=1)
        Vdiff = Vpath[:-1] + Vpath[1:] - Vt[:-1] - Vt[1:]
        safe_Vdiff = jnp.maximum(Vdiff, 1e-300)
        safe_minus_dVt = jnp.maximum(minus_dVt, 1e-300)
        terms = (safe_Vdiff**(0.5*d)) * (ds**d) / (safe_minus_dVt**(d-1))
        bad = jnp.any((Vdiff <= 0) | (minus_dVt <= 0) | (~jnp.isfinite(terms)))
        S = pref*jnp.sum(terms)
        return jnp.where(bad, S + 1e50, S)

    vg = jax.jit(jax.value_and_grad(action))
    x0 = jnp.zeros(case.nphi*nb)
    t0 = time.perf_counter(); v0,g0 = vg(x0); v0.block_until_ready(); g0.block_until_ready(); compile_time = time.perf_counter()-t0
    def fun_jac(x):
        v,g = vg(jnp.asarray(x)); v.block_until_ready(); g.block_until_ready()
        return float(v), np.asarray(g, dtype=float)
    return fun_jac, compile_time


def get_amp_bound(case): return BOUND_FACTOR*np.linalg.norm(case.true_vac-case.false_vac)


def expand_coeffs(old, nphi, old_n, new_n):
    if old is None: return None
    O = np.asarray(old).reshape(nphi, old_n)
    N = np.zeros((nphi, new_n))
    m = min(old_n, new_n); N[:,:m] = O[:,:m]
    return N.ravel()


def candidate_starts(case, nb, prev_coeffs, prev_nb, best_coeffs, best_nb, mode_index):
    starts = []
    nvar = case.nphi*nb; amp = get_amp_bound(case)
    if USE_ZERO_START: starts.append(("zero", np.zeros(nvar)))
    if USE_PREVIOUS_WARM_START and prev_coeffs is not None: starts.append(("warm_previous", expand_coeffs(prev_coeffs, case.nphi, prev_nb, nb)))
    if USE_BEST_WARM_START and best_coeffs is not None: starts.append(("warm_best", expand_coeffs(best_coeffs, case.nphi, best_nb, nb)))
    for i in range(N_RANDOM_STARTS):
        rng = np.random.default_rng(20260601 + 1000*case.nphi + 37*nb + 991*mode_index + 17*i)
        starts.append((f"random_{i+1}", np.clip(rng.normal(scale=RANDOM_SCALE*amp, size=nvar), -amp, amp)))
    return starts


def optimize_one_start(case, nb, fun_jac, x0, label):
    amp = get_amp_bound(case)
    bounds = [(-amp, amp)]*(case.nphi*nb)
    t0 = time.perf_counter()
    res = minimize(fun_jac, x0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": MAXITER, "ftol": FTOL, "gtol": GTOL, "maxls": 50})
    return {"start_label": label, "action": float(res.fun), "coeffs": np.asarray(res.x), "success": bool(res.success), "message": str(res.message), "nit": int(getattr(res,"nit",-1)), "nfev": int(getattr(res,"nfev",-1)), "solve_time_s": time.perf_counter()-t0}


def reconstruct_path_np(case, B, t, coeffs_flat):
    nb = B.shape[0]
    coeffs = np.asarray(coeffs_flat).reshape(case.nphi, nb)
    straight = case.false_vac[None,:] + t[:,None]*(case.true_vac-case.false_vac)[None,:]
    path = straight + B.T @ coeffs.T
    Vpath = V_shift_np(path, case)
    return path, straight, Vpath


def save_mode_npz(case, basis, nb, t, B, coeffs, action, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path, straight, Vpath = reconstruct_path_np(case, B, t, coeffs)
    np.savez(folder/f"mode_{nb:03d}.npz", coeffs=coeffs, basis_matrix=B, t_grid=t, path=path, straight_path=straight, false_vac=case.false_vac, true_vac=case.true_vac, V_path=Vpath, action=action, basis_name=basis, n_basis=nb, nphi=case.nphi, potential_name=case.potential_name, case_label=case.label, action_dim=ACTION_DIM)


def rel_percent(val, ref):
    if ref is None or ref == "": return ""
    return 100*abs(val-ref)/abs(ref)


def adaptive_scan_case_basis(case, basis):
    print("\n"+"="*110); print(f"CASE={case.label}, basis={basis}")
    t = np.linspace(0,1,N_GRID)
    folder = OUTDIR / f"{case.label}__{basis}"
    history = []
    best_action = np.inf; best_coeffs = None; best_nb = None
    prev_coeffs = None; prev_nb = None; prev_action = None
    no_improve = 0; converged = False; stop_reason = "max_basis_reached"
    cum_compile = 0.0; cum_solve = 0.0

    for mode_index, nb in enumerate(range(N_BASIS_START, N_BASIS_MAX+1, N_BASIS_STEP), start=1):
        print("-"*90); print(f"n_basis={nb}")
        B = make_basis_matrix(t, basis, nb)
        fun_jac, compile_time = build_compiled_value_grad(case, B, t)
        cum_compile += compile_time
        starts = candidate_starts(case, nb, prev_coeffs, prev_nb, best_coeffs, best_nb, mode_index)
        results = []
        for label, x0 in starts:
            out = optimize_one_start(case, nb, fun_jac, x0, label)
            results.append(out); cum_solve += out["solve_time_s"]
            print(f"  {label:>13s}: S={out['action']:.10g}, solve={out['solve_time_s']:.4f}s, nit={out['nit']}, ok={out['success']}")
        mode_best = min(results, key=lambda z: z["action"])
        mode_action = mode_best["action"]
        rel_change_prev = "" if prev_action is None or abs(mode_action) == 0 else abs(mode_action-prev_action)/abs(mode_action)
        improved = False; improvement = 0.0
        if mode_action < best_action:
            improvement = np.inf if not np.isfinite(best_action) else (best_action-mode_action)/max(abs(mode_action),1e-300)
            no_improve = 0 if (not np.isfinite(best_action) or improvement > TOLERANCE) else no_improve + 1
            improved = True; best_action = mode_action; best_coeffs = mode_best["coeffs"]; best_nb = nb
        else:
            no_improve += 1
        if SAVE_EVERY_MODE_NPZ: save_mode_npz(case, basis, nb, t, B, mode_best["coeffs"], mode_action, folder)
        ref_key = "S_FB_D3" if ACTION_DIM == 3 else "S_FB_D4"
        ref = case.refs.get(ref_key, None)
        row = {"case_label": case.label, "potential_name": case.potential_name, "action_dim": ACTION_DIM, "nphi": case.nphi, "basis": basis, "n_basis": nb, "mode_best_S": mode_action, "mode_best_start": mode_best["start_label"], "global_best_S_after_mode": best_action, "global_best_n_basis_after_mode": best_nb, "rel_change_from_previous_mode": rel_change_prev, "rel_change_from_previous_mode_percent": "" if rel_change_prev == "" else 100*rel_change_prev, "improved_global_best": improved, "improvement_fraction": "" if not np.isfinite(improvement) else improvement, "improvement_percent": "" if not np.isfinite(improvement) else 100*improvement, "no_improve_count": no_improve, "compile_time_s": compile_time, "solve_time_s_sum": sum(r["solve_time_s"] for r in results), "cumulative_compile_time_s": cum_compile, "cumulative_solve_time_s": cum_solve, "cumulative_total_time_s": cum_compile+cum_solve, "n_starts": len(starts), "success": mode_best["success"], "nit": mode_best["nit"], "nfev": mode_best["nfev"], "message": mode_best["message"], "reference_FB": "" if ref is None else ref, "rel_to_FB_percent": "" if ref is None else rel_percent(best_action, ref), "DeltaV": case.DeltaV}
        history.append(row)
        print(f"  MODE BEST={mode_action:.10g}; GLOBAL BEST={best_action:.10g} at n_basis={best_nb}; no_improve={no_improve}")
        prev_coeffs = mode_best["coeffs"]; prev_nb = nb; prev_action = mode_action
        if no_improve >= PATIENCE:
            converged = True; stop_reason = f"converged: no global-best improvement > {TOLERANCE} for {PATIENCE} basis steps"; print("STOP:", stop_reason); break
    ref_key = "S_FB_D3" if ACTION_DIM == 3 else "S_FB_D4"; ref = case.refs.get(ref_key, None)
    summary = {"case_label": case.label, "potential_name": case.potential_name, "action_dim": ACTION_DIM, "nphi": case.nphi, "basis": basis, "converged": converged, "stop_reason": stop_reason, "best_n_basis": best_nb, "best_S": best_action, "reference_FB": "" if ref is None else ref, "rel_to_FB_percent": "" if ref is None else rel_percent(best_action, ref), "last_n_basis_checked": history[-1]["n_basis"], "n_basis_checked": len(history), "cumulative_compile_time_s": cum_compile, "cumulative_solve_time_s": cum_solve, "cumulative_total_time_s": cum_compile+cum_solve, "DeltaV": case.DeltaV, "false_vac": json.dumps(case.false_vac.tolist()), "true_vac": json.dumps(case.true_vac.tolist()), "output_dir": str(folder)}
    return history, summary


def parse_nphi_list(text):
    """
    Parse field-number lists from terminal.

    Examples
    --------
    "3,5,10"  -> [3, 5, 10]
    "5:20"    -> [5, 6, ..., 20]
    "5:20:2"  -> [5, 7, 9, ..., 19]
    """
    if text is None:
        return None

    text = str(text).strip()
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]

    if ":" in text:
        parts = [int(x.strip()) for x in text.split(":") if x.strip()]
        if len(parts) == 2:
            a, b = parts
            return list(range(a, b + 1))
        if len(parts) == 3:
            a, b, step = parts
            return list(range(a, b + 1, step))
        raise ValueError(f"Could not parse --nphi = {text}")

    return [int(text)]


def apply_cli_args():
    """
    Read terminal options and update the global settings.

    Main examples
    -------------
    Published OptiBounce table, D=3:
        python3 basis_comparison_jax.py --run-mode optibounce_D3

    Same OptiBounce potential but D=4:
        python3 basis_comparison_jax.py --run-mode optibounce_D4

    Mega-random high-dimensional benchmark, D=4:
        python3 basis_comparison_jax.py --run-mode mega_random_D4

    Quick test:
        python3 basis_comparison_jax.py --run-mode optibounce_D3 --nphi 5,10,20 --basis fourier,bspline_local --no-npz
    """
    global POTENTIAL_NAME, ACTION_DIM, NPHI_VALUES, BASIS_LIST, RUN_LABEL, OUTDIR
    global MEGA_RANDOM_CSV, N_BASIS_START, N_BASIS_MAX, N_BASIS_STEP
    global TOLERANCE, PATIENCE, N_GRID, MAXITER, N_RANDOM_STARTS, BOUND_FACTOR
    global SAVE_EVERY_MODE_NPZ

    parser = argparse.ArgumentParser(
        description="Adaptive JAX basis comparison for OptiBounce and mega-random bounce benchmarks."
    )

    parser.add_argument(
        "--run-mode",
        choices=["optibounce_D3", "optibounce_D4", "mega_random_D4"],
        default=None,
        help="Preset choice. Easiest option.",
    )
    parser.add_argument(
        "--potential",
        choices=["optibounce_paper", "mega_random_csv"],
        default=None,
        help="Potential name. Overrides the default/preset if given.",
    )
    parser.add_argument(
        "--dim", "--action-dim",
        dest="action_dim",
        type=int,
        choices=[3, 4],
        default=None,
        help="Action dimension: 3 or 4.",
    )
    parser.add_argument(
        "--nphi",
        default=None,
        help='Field numbers. Examples: "3,5,10", "5:20", "5:20:2".',
    )
    parser.add_argument(
        "--basis",
        default=None,
        help='Comma-separated basis list, e.g. "fourier,bspline_local,hybrid_fourier_bspline".',
    )
    parser.add_argument("--n-basis-start", type=int, default=None)
    parser.add_argument("--n-basis-max", type=int, default=None)
    parser.add_argument("--n-basis-step", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--n-grid", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--n-random-starts", type=int, default=None)
    parser.add_argument("--bound-factor", type=float, default=None)
    parser.add_argument("--mega-random-csv", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--no-npz", action="store_true", help="Do not save per-mode .npz path files.")

    args = parser.parse_args()

    # Presets
    if args.run_mode == "optibounce_D3":
        POTENTIAL_NAME = "optibounce_paper"
        ACTION_DIM = 3
        NPHI_VALUES = list(range(3, 21))
        RUN_LABEL = "optibounce_D3"

    elif args.run_mode == "optibounce_D4":
        POTENTIAL_NAME = "optibounce_paper"
        ACTION_DIM = 4
        NPHI_VALUES = list(range(3, 21))
        RUN_LABEL = "optibounce_D4"

    elif args.run_mode == "mega_random_D4":
        POTENTIAL_NAME = "mega_random_csv"
        ACTION_DIM = 4
        NPHI_VALUES = [5, 10, 20, 30, 40, 50]
        BASIS_LIST = ["fourier", "bspline_local", "hybrid_fourier_bspline"]
        RUN_LABEL = "mega_random_D4"

    # Manual overrides
    if args.potential is not None:
        POTENTIAL_NAME = args.potential

    if args.action_dim is not None:
        ACTION_DIM = args.action_dim

    if args.nphi is not None:
        NPHI_VALUES = parse_nphi_list(args.nphi)

    if args.basis is not None:
        BASIS_LIST = [x.strip() for x in args.basis.split(",") if x.strip()]

    if args.n_basis_start is not None:
        N_BASIS_START = args.n_basis_start
    if args.n_basis_max is not None:
        N_BASIS_MAX = args.n_basis_max
    if args.n_basis_step is not None:
        N_BASIS_STEP = args.n_basis_step
    if args.tolerance is not None:
        TOLERANCE = args.tolerance
    if args.patience is not None:
        PATIENCE = args.patience
    if args.n_grid is not None:
        N_GRID = args.n_grid
    if args.maxiter is not None:
        MAXITER = args.maxiter
    if args.n_random_starts is not None:
        N_RANDOM_STARTS = args.n_random_starts
    if args.bound_factor is not None:
        BOUND_FACTOR = args.bound_factor
    if args.mega_random_csv is not None:
        MEGA_RANDOM_CSV = args.mega_random_csv

    if args.no_npz:
        SAVE_EVERY_MODE_NPZ = False

    if args.outdir is not None:
        OUTDIR = Path(args.outdir)
    else:
        # If run-mode was not used, make a clean automatic label.
        if args.run_mode is None:
            RUN_LABEL = f"{POTENTIAL_NAME}_D{ACTION_DIM}"
        OUTDIR = Path("results") / RUN_LABEL



def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("\nAdaptive JAX basis-comparison engine")
    print("="*100)
    print(f"POTENTIAL_NAME={POTENTIAL_NAME}, ACTION_DIM={ACTION_DIM}, NPHI_VALUES={NPHI_VALUES}")
    print(f"BASIS_LIST={BASIS_LIST}")
    print(f"N_BASIS_START/MAX={N_BASIS_START}/{N_BASIS_MAX}, TOLERANCE/PATIENCE={TOLERANCE}/{PATIENCE}")
    all_history = []; summaries = []
    nphi_list = NPHI_VALUES
    for nphi in nphi_list:
        case = make_case(nphi)
        print("\n"+"#"*110)
        print(f"Built case {case.label}")
        print(f"false={case.false_vac}")
        print(f"true ={case.true_vac}")
        print(f"Vfalse={case.Vfalse:.12g}, Vtrue={case.Vtrue:.12g}, DeltaV={case.DeltaV:.12g}")
        for basis in BASIS_LIST:
            hist, summ = adaptive_scan_case_basis(case, basis)
            all_history.extend(hist); summaries.append(summ)
    dfh = pd.DataFrame(all_history); dfs = pd.DataFrame(summaries)
    hp = OUTDIR/"basis_comparison_history.csv"; sp = OUTDIR/"basis_comparison_summary.csv"
    dfh.to_csv(hp, index=False); dfs.to_csv(sp, index=False)
    print("\n\nFINAL SUMMARY")
    print("="*130)
    cols = ["case_label", "basis", "best_n_basis", "best_S", "reference_FB", "rel_to_FB_percent", "cumulative_total_time_s", "converged"]
    print(dfs[cols].to_string(index=False))
    print(f"\nSaved history: {hp}")
    print(f"Saved summary: {sp}")

if __name__ == "__main__":
    apply_cli_args()
    main()
