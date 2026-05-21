"""
fourier_ct_scan_repeats.py

Purpose
-------
Benchmark whether a low-mode Fourier/JAX initial path improves CosmoTransitions
path deformation compared to straight-line and triangle initial paths.

What it does
------------
1. Builds the same N-field curved-valley benchmark:

   V_N = V_2(h,s) + 1/2 m_spec^2 sum_k [z_k - c_k B(h,s)]^2

2. Runs CosmoTransitions with:
   - straight path
   - triangle path
   - Fourier/JAX path with nmodes in a scan list

3. Repeats each case several times and saves:
   - all raw runs to CSV
   - median summary to CSV
   - a simple mode-scan plot

Run examples
------------
python3 fourier_ct_scan_repeats.py --nfields 5 --modes 1,2,3,5,8 --repeats 5
python3 fourier_ct_scan_repeats.py --nfields 5,10 --modes 1,2,3,5,8 --repeats 5
python3 fourier_ct_scan_repeats.py --nfields 10 --modes 1 --repeats 10 --inspect_once

Packages
--------
pip install numpy scipy matplotlib pandas jax jaxlib cosmoTransitions

Notes
-----
- The action extraction is version-dependent in CosmoTransitions. This script tries hard
  to find it. If action still appears as None, run with --inspect_once and send the output.
- fRatio and deformation steps are parsed from CosmoTransitions' printed output.
"""

import argparse
import io
import re
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cosmoTransitions import pathDeformation as pd_ct


# ============================================================
# 1. Base 2-field benchmark
# ============================================================

LAM = 0.3
EPS = 0.1

# Actual stationary minima for V2 with lambda=0.3, epsilon=0.1.
BASE_TRUE = np.array([0.8942723612839004, 0.7212254614935721], dtype=float)
BASE_FALSE = np.array([-0.6059055312950799, 0.8830215708947969], dtype=float)


def V2_np(h, s):
    return 0.25 * (h*h - 1.0)**2 + 0.25 * (s*s - 1.0)**2 + LAM * h*h * s*s - EPS * h


def dV2_np(h, s):
    dVdh = h * (h*h - 1.0) + 2.0 * LAM * h * s*s - EPS
    dVds = s * (s*s - 1.0) + 2.0 * LAM * s * h*h
    return dVdh, dVds


def V2_jax(h, s):
    return 0.25 * (h*h - 1.0)**2 + 0.25 * (s*s - 1.0)**2 + LAM * h*h * s*s - EPS * h


# ============================================================
# 2. N-field potential
# ============================================================

class NFieldPotential:
    def __init__(self, nfields=5, m_spec=1.0, spectator_amp=0.35):
        if nfields < 2:
            raise ValueError("nfields must be >= 2")
        self.nfields = int(nfields)
        self.m_spec = float(m_spec)
        self.spectator_amp = float(spectator_amp)

        self.phi_true = np.zeros(self.nfields, dtype=float)
        self.phi_false = np.zeros(self.nfields, dtype=float)
        self.phi_true[:2] = BASE_TRUE
        self.phi_false[:2] = BASE_FALSE

        if self.nfields > 2:
            k = np.arange(self.nfields - 2)
            signs = np.where(k % 2 == 0, 1.0, -1.0)
            self.c = self.spectator_amp * signs / np.sqrt(self.nfields - 2)
        else:
            self.c = np.zeros(0, dtype=float)

    def B_np(self, h, s):
        ht = self.phi_true[0]
        hf = self.phi_false[0]
        return (h - ht) * (h - hf) * s

    def dB_np(self, h, s):
        ht = self.phi_true[0]
        hf = self.phi_false[0]
        dBdh = (2.0*h - ht - hf) * s
        dBds = (h - ht) * (h - hf)
        return dBdh, dBds

    def V_np(self, phi):
        phi = np.asarray(phi, dtype=float)
        h = phi[..., 0]
        s = phi[..., 1]
        V = V2_np(h, s)

        if self.nfields > 2:
            z = phi[..., 2:]
            B = self.B_np(h, s)[..., None]
            if z.ndim == 2:
                F = z - self.c[None, :] * B
            else:
                F = z - self.c * B
            V = V + 0.5 * self.m_spec**2 * np.sum(F*F, axis=-1)
        return V

    def dV_np(self, phi):
        phi = np.asarray(phi, dtype=float)
        single = False
        if phi.ndim == 1:
            phi = phi[None, :]
            single = True

        h = phi[:, 0]
        s = phi[:, 1]
        grad = np.zeros_like(phi)
        grad[:, 0], grad[:, 1] = dV2_np(h, s)

        if self.nfields > 2:
            z = phi[:, 2:]
            B = self.B_np(h, s)[:, None]
            dBdh, dBds = self.dB_np(h, s)
            F = z - self.c[None, :] * B
            grad[:, 2:] += self.m_spec**2 * F
            sum_cF = np.sum(self.c[None, :] * F, axis=1)
            grad[:, 0] += -self.m_spec**2 * sum_cF * dBdh
            grad[:, 1] += -self.m_spec**2 * sum_cF * dBds

        return grad[0] if single else grad

    def make_V_jax(self):
        nfields = self.nfields
        m_spec = self.m_spec
        c = jnp.array(self.c)
        ht = float(self.phi_true[0])
        hf = float(self.phi_false[0])

        @jax.jit
        def V_jax(phi):
            h = phi[..., 0]
            s = phi[..., 1]
            V = V2_jax(h, s)
            if nfields > 2:
                z = phi[..., 2:]
                B = ((h - ht) * (h - hf) * s)[..., None]
                F = z - c * B
                V = V + 0.5 * m_spec**2 * jnp.sum(F*F, axis=-1)
            return V

        return V_jax


# ============================================================
# 3. Initial paths
# ============================================================

def sampled_barrier_height(pot, path):
    return float(np.max(pot.V_np(path)) - pot.V_np(pot.phi_false))


def make_straight_path(pot, npts):
    t = np.linspace(0.0, 1.0, npts)
    return (1.0 - t)[:, None] * pot.phi_true + t[:, None] * pot.phi_false


def make_triangle_path(pot, npts, bend_size=0.35):
    mid = 0.5 * (pot.phi_true + pot.phi_false)
    direction2 = pot.phi_false[:2] - pot.phi_true[:2]
    normal2 = np.array([-direction2[1], direction2[0]], dtype=float)
    normal2 = normal2 / np.linalg.norm(normal2)

    candidates = []
    for sign in [1.0, -1.0]:
        bend = mid.copy()
        bend[:2] += sign * bend_size * normal2
        candidates.append(bend)

    # Pick the triangle with a real barrier and the smaller max barrier.
    best_path = None
    best_score = np.inf
    Vfalse = pot.V_np(pot.phi_false)
    for bend in candidates:
        n1 = npts // 2
        n2 = npts - n1
        t1 = np.linspace(0.0, 1.0, n1, endpoint=False)
        t2 = np.linspace(0.0, 1.0, n2)
        part1 = (1.0 - t1)[:, None] * pot.phi_true + t1[:, None] * bend
        part2 = (1.0 - t2)[:, None] * bend + t2[:, None] * pot.phi_false
        path = np.vstack([part1, part2])
        barrier = float(np.max(pot.V_np(path)) - Vfalse)
        if barrier > 1e-8 and barrier < best_score:
            best_score = barrier
            best_path = path
    if best_path is None:
        return make_straight_path(pot, npts)
    return best_path


def make_fourier_path_from_coeffs(pot, coeffs, npts, nmodes):
    coeffs = np.asarray(coeffs, dtype=float).reshape(nmodes, pot.nfields)
    t = np.linspace(0.0, 1.0, npts)
    path = (1.0 - t)[:, None] * pot.phi_true + t[:, None] * pot.phi_false
    for n in range(1, nmodes + 1):
        path += np.sin(n * np.pi * t)[:, None] * coeffs[n - 1][None, :]
    return path


def optimize_fourier_initial_path(pot, npts=120, nmodes=1, maxiter=300, regularizer=1e-3):
    V_jax = pot.make_V_jax()
    nfields = pot.nfields

    t = jnp.linspace(0.0, 1.0, npts)
    phi_true = jnp.array(pot.phi_true)
    phi_false = jnp.array(pot.phi_false)
    V_false = V_jax(phi_false)

    ns = jnp.arange(1, nmodes + 1)
    basis = jnp.sin(jnp.pi * t[:, None] * ns[None, :])
    straight = (1.0 - t)[:, None] * phi_true + t[:, None] * phi_false

    @jax.jit
    def objective(flat_coeffs):
        coeffs = flat_coeffs.reshape((nmodes, nfields))
        path = straight + basis @ coeffs

        dpath = path[1:] - path[:-1]
        ds = jnp.sqrt(jnp.sum(dpath*dpath, axis=1) + 1e-30)
        mid = 0.5 * (path[1:] + path[:-1])
        Vmid = V_jax(mid)

        barrier = jnp.maximum(Vmid - V_false, 1e-10)
        tension_like = jnp.sum(ds * jnp.sqrt(2.0 * barrier))

        second = path[2:] - 2.0 * path[1:-1] + path[:-2]
        smooth = jnp.sum(second*second)

        box = jnp.sum(jnp.maximum(jnp.abs(path) - 3.0, 0.0)**2)

        maxV = jnp.max(V_jax(path))
        no_barrier_penalty = jnp.maximum(1e-4 - (maxV - V_false), 0.0)**2

        return tension_like + regularizer * smooth + 10.0 * box + 1e4 * no_barrier_penalty

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_fun(x):
        val, grad = value_and_grad(jnp.array(x))
        return float(val), np.array(grad, dtype=float)

    x0 = np.zeros((nmodes, nfields), dtype=float).ravel()
    _ = scipy_fun(x0)  # JIT warm-up

    t0 = time.perf_counter()
    res = minimize(
        fun=lambda x: scipy_fun(x)[0],
        x0=x0,
        jac=lambda x: scipy_fun(x)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    t1 = time.perf_counter()

    path = make_fourier_path_from_coeffs(pot, res.x, npts=npts, nmodes=nmodes)
    info = {
        "fourier_success": bool(res.success),
        "fourier_message": str(res.message),
        "fourier_objective": float(res.fun),
        "fourier_nit": int(res.nit),
        "prep_time_sec": t1 - t0,
        "sampled_barrier": sampled_barrier_height(pot, path),
    }
    return path, info


# ============================================================
# 4. CosmoTransitions output parsing / inspection
# ============================================================

@dataclass
class RunResult:
    nfields: int
    nmodes: int | str
    repeat: int
    method: str
    success: bool
    action: float | None
    fRatio: float | None
    steps_total: int | None
    n_deformations: int | None
    prep_time_sec: float
    ct_time_sec: float
    total_time_sec: float
    sampled_barrier: float | None
    error_short: str | None


def extract_from_ct_stdout(text):
    matches = re.findall(
        r"Path deformation converged\.\s*(\d+)\s*steps\.\s*fRatio\s*=\s*([0-9.eE+-]+)",
        text,
    )
    if not matches:
        return None, None, None
    steps = [int(m[0]) for m in matches]
    frs = [float(m[1]) for m in matches]
    return frs[-1], int(np.sum(steps)), len(matches)


def _safe_float(x):
    try:
        y = float(x)
        if np.isfinite(y):
            return y
    except Exception:
        pass
    return None


def extract_action_from_output(out: Any):
    """Best-effort action extraction across CosmoTransitions versions."""
    seen = set()
    action_names = {"action", "S", "SE", "S_E", "action1D", "action2D", "tunneling_action"}

    def rec(obj, depth=0):
        if obj is None or depth > 5:
            return None
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        # Direct numeric is not enough; avoid returning arbitrary numbers.
        if isinstance(obj, dict):
            # First check likely keys.
            for k, v in obj.items():
                if str(k) in action_names or "action" in str(k).lower():
                    val = _safe_float(v)
                    if val is not None:
                        return val
                    # Sometimes action is a zero-arg method.
                    if callable(v):
                        try:
                            val = _safe_float(v())
                            if val is not None:
                                return val
                        except Exception:
                            pass
            # Then search likely nested containers.
            for k, v in obj.items():
                if any(word in str(k).lower() for word in ["profile", "tunnel", "instant", "bounce"]):
                    val = rec(v, depth + 1)
                    if val is not None:
                        return val
            return None

        # Object attributes and methods.
        for name in ["action", "S", "SE", "S_E", "findAction", "action1D", "action2D"]:
            if hasattr(obj, name):
                try:
                    attr = getattr(obj, name)
                    if callable(attr):
                        val = _safe_float(attr())
                    else:
                        val = _safe_float(attr)
                    if val is not None:
                        return val
                except Exception:
                    pass

        # Some objects expose __dict__.
        if hasattr(obj, "__dict__"):
            return rec(vars(obj), depth + 1)

        return None

    return rec(out)


def inspect_ct_output(out: Any, max_depth=2):
    """Print structure of CT output once, to find where action lives."""
    seen = set()

    def rec(obj, indent=0, depth=0):
        pad = "  " * indent
        if obj is None:
            print(pad + "None")
            return
        if id(obj) in seen:
            print(pad + f"<seen {type(obj)}>" )
            return
        seen.add(id(obj))

        print(pad + f"type={type(obj)}")
        if isinstance(obj, dict):
            print(pad + "keys:", list(obj.keys()))
            if depth < max_depth:
                for k, v in obj.items():
                    print(pad + f"[{k!r}]")
                    rec(v, indent + 1, depth + 1)
        elif hasattr(obj, "__dict__"):
            keys = list(vars(obj).keys())
            print(pad + "attrs:", keys[:40])
            if depth < max_depth:
                for k in keys[:20]:
                    v = getattr(obj, k)
                    if isinstance(v, (int, float, str, list, tuple, dict, np.ndarray)) or hasattr(v, "__dict__"):
                        print(pad + f".{k}")
                        rec(v, indent + 1, depth + 1)
        else:
            s = repr(obj)
            print(pad + s[:200])

    rec(out)


def run_ct_with_path(pot, path, prep_time=0.0, verbose=False, inspect=False):
    t0 = time.perf_counter()
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = pd_ct.fullTunneling(
                path,
                pot.V_np,
                pot.dV_np,
                maxiter=40,
                verbose=verbose,
            )
        ct_log = buf.getvalue()
        t1 = time.perf_counter()

        if inspect:
            print("\n===== CT PRINTED LOG =====")
            print(ct_log)
            print("===== CT OUTPUT STRUCTURE =====")
            inspect_ct_output(out)
            print("===== END INSPECTION =====\n")

        fRatio, steps_total, n_deformations = extract_from_ct_stdout(ct_log)
        action = extract_action_from_output(out)
        return True, action, fRatio, steps_total, n_deformations, t1 - t0, None
    except Exception as e:
        t1 = time.perf_counter()
        short = str(e).split("\n")[-1]
        return False, None, None, None, None, t1 - t0, short + "\n" + traceback.format_exc(limit=2)


# ============================================================
# 5. One run, scan, summaries
# ============================================================

def run_one_case(pot, method, nmodes, repeat, npts, maxiter, inspect=False):
    prep0 = time.perf_counter()
    extra = {}

    if method == "straight":
        path = make_straight_path(pot, npts)
        prep_time = time.perf_counter() - prep0
    elif method == "triangle":
        path = make_triangle_path(pot, npts)
        prep_time = time.perf_counter() - prep0
    elif method == "fourier_jax":
        path, info = optimize_fourier_initial_path(pot, npts=npts, nmodes=int(nmodes), maxiter=maxiter)
        prep_time = info["prep_time_sec"]
        extra.update(info)
    else:
        raise ValueError(f"unknown method {method}")

    barrier = sampled_barrier_height(pot, path)
    success, action, fRatio, steps_total, n_deformations, ct_time, err = run_ct_with_path(
        pot, path, prep_time=prep_time, verbose=False, inspect=inspect,
    )

    result = RunResult(
        nfields=pot.nfields,
        nmodes=nmodes,
        repeat=repeat,
        method=method,
        success=success,
        action=action,
        fRatio=fRatio,
        steps_total=steps_total,
        n_deformations=n_deformations,
        prep_time_sec=prep_time,
        ct_time_sec=ct_time,
        total_time_sec=prep_time + ct_time,
        sampled_barrier=barrier,
        error_short=err,
    )
    row = asdict(result)
    row.update(extra)
    return row


def median_or_nan(x):
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan
    return float(np.median(arr))


def summarize(df):
    rows = []
    group_cols = ["nfields", "nmodes", "method"]
    for keys, g in df.groupby(group_cols, dropna=False):
        nfields, nmodes, method = keys
        ok = g[g["success"] == True]
        row = {
            "nfields": nfields,
            "nmodes": nmodes,
            "method": method,
            "success_count": int(g["success"].sum()),
            "n_runs": int(len(g)),
            "action_median": median_or_nan(ok["action"]) if len(ok) else np.nan,
            "fRatio_median": median_or_nan(ok["fRatio"]) if len(ok) else np.nan,
            "steps_median": median_or_nan(ok["steps_total"]) if len(ok) else np.nan,
            "prep_median": median_or_nan(ok["prep_time_sec"]) if len(ok) else np.nan,
            "ct_median": median_or_nan(ok["ct_time_sec"]) if len(ok) else np.nan,
            "total_median": median_or_nan(ok["total_time_sec"]) if len(ok) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["nfields", "nmodes", "method"])


def print_summary(summary):
    print("\n================ MEDIAN SUMMARY ================")
    cols = ["nfields", "nmodes", "method", "success_count", "n_runs", "action_median", "fRatio_median", "steps_median", "prep_median", "ct_median", "total_median"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    # Show direct straight vs Fourier improvements where available.
    print("\n================ FOURIER VS STRAIGHT ================")
    for nfields in sorted(summary["nfields"].unique()):
        srow = summary[(summary.nfields == nfields) & (summary.method == "straight")]
        if len(srow) == 0:
            continue
        srow = srow.iloc[0]
        for _, frow in summary[(summary.nfields == nfields) & (summary.method == "fourier_jax")].iterrows():
            if pd.notna(srow.steps_median) and pd.notna(frow.steps_median):
                step_red = 100.0 * (srow.steps_median - frow.steps_median) / srow.steps_median
            else:
                step_red = np.nan
            if pd.notna(srow.total_median) and pd.notna(frow.total_median):
                total_red = 100.0 * (srow.total_median - frow.total_median) / srow.total_median
            else:
                total_red = np.nan
            if pd.notna(srow.fRatio_median) and pd.notna(frow.fRatio_median):
                fr_factor = srow.fRatio_median / frow.fRatio_median
            else:
                fr_factor = np.nan
            print(
                f"N={nfields}, modes={frow.nmodes}: "
                f"step reduction={step_red:.2f}%, "
                f"total time reduction={total_red:.2f}%, "
                f"fRatio improvement factor={fr_factor:.3g}"
            )


def plot_mode_scan(summary, outprefix):
    for nfields in sorted(summary["nfields"].unique()):
        sub = summary[summary["nfields"] == nfields]
        straight = sub[sub["method"] == "straight"]
        fourier = sub[sub["method"] == "fourier_jax"].copy()
        if len(fourier) == 0:
            continue
        fourier["nmodes_num"] = pd.to_numeric(fourier["nmodes"], errors="coerce")
        fourier = fourier.sort_values("nmodes_num")

        # Total time plot
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(fourier["nmodes_num"], fourier["total_median"], marker="o", label="Fourier total")
        if len(straight) > 0:
            plt.axhline(float(straight["total_median"].iloc[0]), linestyle="--", label="Straight total")
        plt.xlabel("Fourier modes")
        plt.ylabel("median total time [s]")
        plt.title(f"Total time vs modes, N={nfields}")
        plt.legend()
        plt.tight_layout()
        fname = f"{outprefix}_time_N{nfields}.png"
        plt.savefig(fname, dpi=200)
        print(f"Saved {fname}")

        # Steps plot
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(fourier["nmodes_num"], fourier["steps_median"], marker="o", label="Fourier steps")
        if len(straight) > 0:
            plt.axhline(float(straight["steps_median"].iloc[0]), linestyle="--", label="Straight steps")
        plt.xlabel("Fourier modes")
        plt.ylabel("median CT deformation steps")
        plt.title(f"Deformation steps vs modes, N={nfields}")
        plt.legend()
        plt.tight_layout()
        fname = f"{outprefix}_steps_N{nfields}.png"
        plt.savefig(fname, dpi=200)
        print(f"Saved {fname}")

        # fRatio plot
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(fourier["nmodes_num"], fourier["fRatio_median"], marker="o", label="Fourier fRatio")
        if len(straight) > 0:
            plt.axhline(float(straight["fRatio_median"].iloc[0]), linestyle="--", label="Straight fRatio")
        plt.xlabel("Fourier modes")
        plt.ylabel("median final fRatio")
        plt.title(f"Final fRatio vs modes, N={nfields}")
        plt.legend()
        plt.tight_layout()
        fname = f"{outprefix}_fratio_N{nfields}.png"
        plt.savefig(fname, dpi=200)
        print(f"Saved {fname}")


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfields", type=str, default="5,10", help="comma list, e.g. 5,10")
    parser.add_argument("--modes", type=str, default="1,2,3,5,8", help="comma list of Fourier modes")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--npts", type=int, default=120)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--m_spec", type=float, default=1.0)
    parser.add_argument("--spectator_amp", type=float, default=0.35)
    parser.add_argument("--outprefix", type=str, default="scan")
    parser.add_argument("--include_triangle", action="store_true", help="Include triangle baseline. Default skips it to save time.")
    parser.add_argument("--inspect_once", action="store_true", help="Print CT output structure once to debug action extraction.")
    args = parser.parse_args()

    nfields_list = parse_int_list(args.nfields)
    modes_list = parse_int_list(args.modes)
    methods_base = ["straight"]
    if args.include_triangle:
        methods_base.append("triangle")

    print("Fourier-preconditioned CosmoTransitions scan")
    print(f"N list       = {nfields_list}")
    print(f"modes list   = {modes_list}")
    print(f"repeats      = {args.repeats}")
    print(f"npts         = {args.npts}")
    print(f"m_spec       = {args.m_spec}")
    print(f"spectator_amp= {args.spectator_amp}")
    print()

    rows = []
    inspected = False

    for N in nfields_list:
        pot = NFieldPotential(nfields=N, m_spec=args.m_spec, spectator_amp=args.spectator_amp)
        print("\n================================================")
        print(f"N={N}")
        print(f"true  V={pot.V_np(pot.phi_true):.10f}, |grad|={np.linalg.norm(pot.dV_np(pot.phi_true)):.3e}")
        print(f"false V={pot.V_np(pot.phi_false):.10f}, |grad|={np.linalg.norm(pot.dV_np(pot.phi_false)):.3e}")
        print("================================================")

        for rep in range(1, args.repeats + 1):
            print(f"\n--- repeat {rep}/{args.repeats}, N={N} ---")

            # Baselines only need nmodes='baseline'. We repeat for timing statistics.
            for method in methods_base:
                inspect = args.inspect_once and not inspected
                row = run_one_case(pot, method, "baseline", rep, args.npts, args.maxiter, inspect=inspect)
                inspected = inspected or inspect
                rows.append(row)
                print(
                    f"{method:<12} ok={row['success']} "
                    f"fRatio={row['fRatio']} steps={row['steps_total']} "
                    f"prep={row['prep_time_sec']:.4f}s CT={row['ct_time_sec']:.4f}s total={row['total_time_sec']:.4f}s action={row['action']}"
                )

            # Fourier mode scan.
            for m in modes_list:
                inspect = args.inspect_once and not inspected
                row = run_one_case(pot, "fourier_jax", m, rep, args.npts, args.maxiter, inspect=inspect)
                inspected = inspected or inspect
                rows.append(row)
                print(
                    f"fourier m={m:<3} ok={row['success']} "
                    f"fRatio={row['fRatio']} steps={row['steps_total']} "
                    f"prep={row['prep_time_sec']:.4f}s CT={row['ct_time_sec']:.4f}s total={row['total_time_sec']:.4f}s action={row['action']}"
                )
                if not row["success"]:
                    print("  error:", row["error_short"])

    raw = pd.DataFrame(rows)
    raw_csv = f"{args.outprefix}_raw.csv"
    raw.to_csv(raw_csv, index=False)
    print(f"\nSaved {raw_csv}")

    summary = summarize(raw)
    summary_csv = f"{args.outprefix}_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")

    print_summary(summary)
    plot_mode_scan(summary, args.outprefix)

    print("\nDone. Send me:")
    print(f"  1. {summary_csv}")
    print(f"  2. terminal MEDIAN SUMMARY")
    print(f"  3. plots {args.outprefix}_time_N*.png, {args.outprefix}_steps_N*.png, {args.outprefix}_fratio_N*.png")
    print("If action_median is still NaN, rerun one short case with --inspect_once and send me that output.")


if __name__ == "__main__":
    main()
