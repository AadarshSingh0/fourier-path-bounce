#!/usr/bin/env python3
"""
Uniform adaptive JAX-Fourier solver and plotting script for the OptiBounce/Table-I benchmark in DIMENSION 3.

This is the version to compare with the published OptiBounce table whose actions
are ~200--400. Those are Dimension -> 3 / O(3)-type actions, not the Dimension -> 4
actions we previously computed.

Potential:
    V(phi) = (sum_i c_i (phi_i - 1)^2 - delta) * sum_i phi_i^2

Fourier path:
    phi_i(t) = phi_false_i + t(phi_true_i - phi_false_i)
               + sum_k a_ik sin(k*pi*t)

Reduced tunneling-potential action in dimension d:
    S_d = c_d ∫ ds (V - V_t)^(d/2) / (-dV_t/ds)^(d-1)

where
    c_d = (d-1)^(d-1) * Omega_d / d,
    Omega_d = 2*pi^(d/2)/Gamma(d/2).

For d=4:
    c_4 = 27*pi^2/2.
For d=3:
    c_3 = 16*pi/3.

Discrete version:
    S_d ≈ c_d Σ [(Vbar - Vtbar)^(d/2) * ds^d / (-dVt)^(d-1)].

Run:
    source cosmo_env/bin/activate
    python3 jax_fourier_optibounce_table_D3.py

Outputs:
    jax_fourier_optibounce_D3_history.csv
    jax_fourier_optibounce_D3_summary.csv
    plots/optibounce_table_D3_action_vs_nphi.png/pdf
    plots/optibounce_table_D3_time_vs_nphi.png/pdf
    plots/optibounce_table_D3_action_deviation_vs_nphi.png/pdf
"""

import time
from dataclasses import dataclass
from typing import Optional
from math import gamma

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


ACTION_DIM = 3

# Published OptiBounce Table-I reference data are available for Nphi = 3,...,20.
# Nphi=2 is not included in the published table, so the default scan starts at 3.
NPHI_VALUES = list(range(3, 21))

N_START = 1
N_MAX = 12
N_STEP = 1

TOLERANCE = 1.0e-2
PATIENCE = 3

N_GRID = 260
MAXITER = 800
GTOL = 1e-7
FTOL = 1e-10

USE_ZERO_START = True
USE_PREVIOUS_WARM_START = True
USE_BEST_WARM_START = True
N_RANDOM_STARTS = 3
RANDOM_SCALE = 0.25

USE_GLOBAL_RESCUE_FOR_N4 = False

BOUND_FACTOR = 1.0

HISTORY_CSV = "jax_fourier_optibounce_D3_history.csv"
SUMMARY_CSV = "jax_fourier_optibounce_D3_summary.csv"
PLOT_DIR = "plots"


OPTIBOUNCE_DATA = {
    3: {"delta": 0.065, "c": [0.684373, 0.181928, 0.295089], "S_OB": 240.049, "t_setup_OB": 0.191, "t_sol_OB": 0.009, "S_FB": 240.403, "t_FB": 0.294, "S_CT": 240.324, "t_CT": 0.492},
    4: {"delta": 0.11, "c": [0.534808, 0.77023, 0.838912, 0.00517238], "S_OB": 227.023, "t_setup_OB": 0.217, "t_sol_OB": 0.081, "S_FB": 230.864, "t_FB": 0.615, "S_CT": 230.397, "t_CT": 3.223},
    5: {"delta": 0.13, "c": [0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "S_OB": 233.523, "t_setup_OB": 0.259, "t_sol_OB": 0.025, "S_FB": 233.716, "t_FB": 0.408, "S_CT": 233.357, "t_CT": 0.976},
    6: {"delta": 0.15, "c": [0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "S_OB": 270.363, "t_setup_OB": 0.300, "t_sol_OB": 0.031, "S_FB": 270.578, "t_FB": 0.442, "S_CT": 271.769, "t_CT": 3.917},
    7: {"delta": 0.2, "c": [0.5233, 0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "S_OB": 250.054, "t_setup_OB": 0.355, "t_sol_OB": 0.046, "S_FB": 250.222, "t_FB": 0.482, "S_CT": 249.845, "t_CT": 3.802},
    8: {"delta": 0.22, "c": [0.2434, 0.5233, 0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.51723], "S_OB": 268.259, "t_setup_OB": 0.404, "t_sol_OB": 0.085, "S_FB": 268.486, "t_FB": 0.606, "S_CT": 269.368, "t_CT": 4.143},
    9: {"delta": 0.29, "c": [0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "S_OB": 204.609, "t_setup_OB": 0.452, "t_sol_OB": 0.083, "S_FB": 204.796, "t_FB": 0.675, "S_CT": 205.888, "t_CT": 0.769},
    10: {"delta": 0.27, "c": [0.12, 0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "S_OB": 261.468, "t_setup_OB": 0.507, "t_sol_OB": 0.088, "S_FB": 261.703, "t_FB": 0.779, "S_CT": 261.273, "t_CT": 1.279},
    11: {"delta": 0.30, "c": [0.23, 0.21, 0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "S_OB": 273.271, "t_setup_OB": 0.565, "t_sol_OB": 0.116, "S_FB": 273.564, "t_FB": 0.861, "S_CT": None, "t_CT": None},
    12: {"delta": 0.32, "c": [0.12, 0.11, 0.12, 0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "S_OB": 249.691, "t_setup_OB": 0.618, "t_sol_OB": 0.158, "S_FB": 249.928, "t_FB": 0.950, "S_CT": None, "t_CT": None},
    13: {"delta": 0.39, "c": [0.54, 0.47, 0.53, 0.28, 0.35, 0.27, 0.42, 0.59, 0.33, 0.16, 0.38, 0.35, 0.17], "S_OB": 293.383, "t_setup_OB": 0.671, "t_sol_OB": 0.153, "S_FB": 293.653, "t_FB": 1.012, "S_CT": None, "t_CT": None},
    14: {"delta": 0.39, "c": [0.39, 0.23, 0.26, 0.40, 0.11, 0.42, 0.41, 0.27, 0.42, 0.54, 0.18, 0.59, 0.13, 0.29], "S_OB": 294.677, "t_setup_OB": 0.731, "t_sol_OB": 0.528, "S_FB": 294.877, "t_FB": 1.114, "S_CT": None, "t_CT": None},
    15: {"delta": 0.42, "c": [0.21, 0.22, 0.22, 0.23, 0.39, 0.55, 0.43, 0.12, 0.16, 0.58, 0.25, 0.50, 0.45, 0.35, 0.45], "S_OB": 312.760, "t_setup_OB": 0.816, "t_sol_OB": 0.455, "S_FB": 313.039, "t_FB": 1.222, "S_CT": None, "t_CT": None},
    16: {"delta": 0.45, "c": [0.42, 0.34, 0.43, 0.22, 0.59, 0.41, 0.58, 0.41, 0.26, 0.45, 0.16, 0.31, 0.39, 0.57, 0.43, 0.10], "S_OB": 366.870, "t_setup_OB": 0.881, "t_sol_OB": 0.471, "S_FB": 366.978, "t_FB": 1.885, "S_CT": None, "t_CT": None},
    17: {"delta": 0.52, "c": [0.24, 0.35, 0.39, 0.56, 0.37, 0.41, 0.52, 0.31, 0.52, 0.22, 0.58, 0.39, 0.39, 0.17, 0.46, 0.30, 0.37], "S_OB": 342.537, "t_setup_OB": 0.961, "t_sol_OB": 0.734, "S_FB": 342.893, "t_FB": 1.395, "S_CT": None, "t_CT": None},
    18: {"delta": 0.47, "c": [0.18, 0.17, 0.30, 0.22, 0.38, 0.48, 0.11, 0.49, 0.43, 0.47, 0.21, 0.29, 0.32, 0.36, 0.30, 0.56, 0.46, 0.42], "S_OB": 390.925, "t_setup_OB": 1.060, "t_sol_OB": 0.772, "S_FB": 391.231, "t_FB": 1.544, "S_CT": None, "t_CT": None},
    19: {"delta": 0.56, "c": [0.40, 0.14, 0.10, 0.43, 0.39, 0.27, 0.33, 0.59, 0.48, 0.36, 0.24, 0.28, 0.51, 0.59, 0.40, 0.39, 0.24, 0.35, 0.20], "S_OB": 339.287, "t_setup_OB": 1.116, "t_sol_OB": 1.537, "S_FB": 339.467, "t_FB": 2.436, "S_CT": None, "t_CT": None},
    20: {"delta": 0.55, "c": [0.42, 0.11, 0.47, 0.13, 0.16, 0.24, 0.58, 0.53, 0.38, 0.44, 0.18, 0.46, 0.47, 0.27, 0.53, 0.24, 0.33, 0.40, 0.32, 0.29], "S_OB": 381.886, "t_setup_OB": 1.178, "t_sol_OB": 1.309, "S_FB": 382.153, "t_FB": 1.864, "S_CT": None, "t_CT": None},
}


@dataclass
class CaseData:
    nphi: int
    c: np.ndarray
    delta: float
    false_vac: np.ndarray
    true_vac: np.ndarray
    Vfalse: float
    Vtrue: float
    DeltaV: float
    refs: dict


def V_np(phi, c, delta):
    phi = np.asarray(phi, dtype=float)
    A = np.sum(c * (phi - 1.0)**2, axis=-1) - delta
    B = np.sum(phi**2, axis=-1)
    return A * B


def grad_np(phi, c, delta):
    phi = np.asarray(phi, dtype=float)
    A = np.sum(c * (phi - 1.0)**2) - delta
    B = np.sum(phi**2)
    return 2.0 * c * (phi - 1.0) * B + 2.0 * phi * A


def make_case(nphi):
    data = OPTIBOUNCE_DATA[nphi]
    c = np.array(data["c"], dtype=float)
    delta = float(data["delta"])
    false = np.zeros(nphi)

    def obj(x):
        return float(V_np(x, c, delta))

    def jac(x):
        return grad_np(x, c, delta)

    res = minimize(obj, np.ones(nphi), jac=jac, method="BFGS",
                   options={"gtol": 1e-12, "maxiter": 5000})

    true = res.x.astype(float)

    Vfalse = float(obj(false))
    Vtrue = float(obj(true))
    DeltaV = Vfalse - Vtrue

    if DeltaV <= 0:
        raise RuntimeError(f"Bad vacuum ordering for nphi={nphi}: DeltaV={DeltaV}")

    return CaseData(nphi, c, delta, false, true, Vfalse, Vtrue, DeltaV, data)


def omega_d(d):
    return 2.0 * np.pi ** (0.5 * d) / gamma(0.5 * d)


def action_prefactor(d):
    return ((d - 1.0) ** (d - 1.0)) * omega_d(d) / d


def fourier_basis_np(t, n_modes):
    return np.array([np.sin((k + 1) * np.pi * t) for k in range(n_modes)], dtype=float)


def expand_coeffs(old_coeffs, nphi, old_modes, new_modes):
    if old_coeffs is None:
        return None
    old = np.asarray(old_coeffs, dtype=float).reshape(nphi, old_modes)
    new = np.zeros((nphi, new_modes), dtype=float)
    m = min(old_modes, new_modes)
    new[:, :m] = old[:, :m]
    return new.ravel()


def get_amp_bound(case):
    return BOUND_FACTOR * np.linalg.norm(case.true_vac - case.false_vac)


def build_compiled_value_grad(case, n_modes, n_grid, action_dim):
    t_np = np.linspace(0.0, 1.0, n_grid)
    t = jnp.asarray(t_np)
    B = jnp.asarray(fourier_basis_np(t_np, n_modes))

    c = jnp.asarray(case.c)
    delta = jnp.asarray(case.delta)
    false = jnp.asarray(case.false_vac)
    true = jnp.asarray(case.true_vac)
    Vfalse = jnp.asarray(case.Vfalse)
    Vtrue = jnp.asarray(case.Vtrue)
    nphi = case.nphi

    d = float(action_dim)
    pref = jnp.asarray(action_prefactor(d))

    def V_shift(phi):
        A = jnp.sum(c * (phi - 1.0) ** 2, axis=-1) - delta
        R2 = jnp.sum(phi ** 2, axis=-1)
        return A * R2 - Vfalse

    def action(coeffs_flat):
        coeffs = coeffs_flat.reshape((nphi, n_modes))
        base = false[None, :] + t[:, None] * (true - false)[None, :]
        phi = base + B.T @ coeffs.T

        Vphi = V_shift(phi)
        Vt = Vfalse + t * t * (3.0 - 2.0 * t) * (Vtrue - Vfalse)

        dVt = Vt[1:] - Vt[:-1]
        minus_dVt = -dVt

        dphi = phi[1:] - phi[:-1]
        ds = jnp.linalg.norm(dphi, axis=1)

        # Keep the same "sum of endpoint values" convention as our previous O(4) code.
        Vdiff = Vphi[:-1] + Vphi[1:] - Vt[:-1] - Vt[1:]

        safe_Vdiff = jnp.maximum(Vdiff, 1e-300)
        safe_minus_dVt = jnp.maximum(minus_dVt, 1e-300)

        terms = (safe_Vdiff ** (0.5 * d)) * (ds ** d) / (safe_minus_dVt ** (d - 1.0))
        bad = jnp.any((Vdiff <= 0.0) | (minus_dVt <= 0.0) | (~jnp.isfinite(terms)))

        S = pref * jnp.sum(terms)
        return jnp.where(bad, S + 1e50, S)

    value_grad = jax.jit(jax.value_and_grad(action))

    x_compile = jnp.zeros(case.nphi * n_modes)
    t0 = time.perf_counter()
    v0, g0 = value_grad(x_compile)
    v0.block_until_ready()
    g0.block_until_ready()
    compile_time = time.perf_counter() - t0

    def fun_jac(x):
        v, g = value_grad(jnp.asarray(x))
        v.block_until_ready()
        g.block_until_ready()
        return float(v), np.asarray(g, dtype=float)

    return fun_jac, compile_time


def optimize_with_compiled_fun(case, n_modes, fun_jac, x0, start_label):
    """
    Uniform local optimization for every N_phi.

    This version deliberately does NOT apply any special rescue to N_phi=4
    or to any other individual case. Every case is treated identically.

    Robustness comes only from using the same allowed starts for all cases:
      - zero start
      - warm start from previous mode
      - warm start from best previous mode
      - deterministic random starts
    """
    nvar = case.nphi * n_modes
    if x0 is None:
        x0 = np.zeros(nvar, dtype=float)

    amp_bound = get_amp_bound(case)
    bounds = [(-amp_bound, amp_bound)] * nvar

    t0 = time.perf_counter()

    res = minimize(
        fun_jac,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": MAXITER,
            "ftol": FTOL,
            "gtol": GTOL,
            "maxls": 50,
        },
    )

    solve_time = time.perf_counter() - t0

    return {
        "start_label": start_label,
        "action": float(res.fun),
        "coeffs": np.asarray(res.x),
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(getattr(res, "nfev", -1)),
        "njev": int(getattr(res, "njev", -1)),
        "nit": int(getattr(res, "nit", -1)),
        "solve_time_s": solve_time,
    }


def candidate_starts(case, n_modes, previous_coeffs, previous_modes, best_coeffs, best_modes, mode_index):
    starts = []
    nvar = case.nphi * n_modes
    amp_bound = get_amp_bound(case)

    if USE_ZERO_START:
        starts.append(("zero", np.zeros(nvar, dtype=float)))

    if USE_PREVIOUS_WARM_START and previous_coeffs is not None and previous_modes is not None:
        starts.append(("warm_previous", expand_coeffs(previous_coeffs, case.nphi, previous_modes, n_modes)))

    if USE_BEST_WARM_START and best_coeffs is not None and best_modes is not None:
        starts.append(("warm_best", expand_coeffs(best_coeffs, case.nphi, best_modes, n_modes)))

    for i in range(N_RANDOM_STARTS):
        seed = 314159 + 1000 * case.nphi + 37 * n_modes + 17 * i + 991 * mode_index
        rng = np.random.default_rng(seed)
        x0 = rng.normal(scale=RANDOM_SCALE * amp_bound, size=nvar)
        x0 = np.clip(x0, -amp_bound, amp_bound)
        starts.append((f"random_{i+1}", x0))

    return starts


def rel_percent(S, ref):
    if ref is None:
        return ""
    return 100.0 * abs(S - ref) / abs(ref)


def adaptive_scan_case(case):
    history_rows = []

    best_action = np.inf
    best_coeffs = None
    best_modes = None

    previous_coeffs = None
    previous_modes = None
    previous_mode_action = None

    no_improve_count = 0
    converged = False
    stop_reason = "max_modes_reached"

    cumulative_compile = 0.0
    cumulative_solve = 0.0
    mode_index = 0

    for n_modes in range(N_START, N_MAX + 1, N_STEP):
        mode_index += 1

        print("\n" + "-" * 88)
        print(f"nphi={case.nphi}, n_modes={n_modes}")

        fun_jac, compile_time = build_compiled_value_grad(case, n_modes, N_GRID, ACTION_DIM)
        cumulative_compile += compile_time

        starts = candidate_starts(case, n_modes, previous_coeffs, previous_modes, best_coeffs, best_modes, mode_index)

        run_results = []
        for start_label, x0 in starts:
            out = optimize_with_compiled_fun(case, n_modes, fun_jac, x0, start_label)
            run_results.append(out)
            cumulative_solve += out["solve_time_s"]

            print(f"  start={start_label:>13s}: S_D={out['action']:.10g}, solve={out['solve_time_s']:.4f}s, success={out['success']}, nit={out['nit']}")

        mode_best = min(run_results, key=lambda r: r["action"])
        mode_action = mode_best["action"]
        mode_coeffs = mode_best["coeffs"]

        rel_change_prev = ""
        if previous_mode_action is not None and abs(mode_action) > 0:
            rel_change_prev = abs(mode_action - previous_mode_action) / abs(mode_action)

        improved = False
        improvement = 0.0

        if mode_action < best_action:
            if np.isfinite(best_action):
                improvement = (best_action - mode_action) / max(abs(mode_action), 1e-300)
            else:
                improvement = np.inf

            if (not np.isfinite(best_action)) or improvement > TOLERANCE:
                no_improve_count = 0
            else:
                no_improve_count += 1

            improved = True
            best_action = mode_action
            best_coeffs = mode_coeffs
            best_modes = n_modes
        else:
            no_improve_count += 1

        refs = case.refs

        row = {
            "nphi": case.nphi,
            "delta": case.delta,
            "action_dim": ACTION_DIM,
            "n_modes": n_modes,
            "mode_best_start": mode_best["start_label"],
            "mode_best_S": mode_action,
            "rel_change_from_previous_mode": rel_change_prev,
            "rel_change_from_previous_mode_percent": "" if rel_change_prev == "" else 100.0 * rel_change_prev,
            "global_best_S_after_mode": best_action,
            "global_best_n_modes_after_mode": best_modes,
            "improved_global_best": improved,
            "improvement_fraction": improvement if np.isfinite(improvement) else "",
            "improvement_percent": "" if not np.isfinite(improvement) else 100.0 * improvement,
            "no_improve_count": no_improve_count,
            "mode_compile_time_s": compile_time,
            "mode_solve_time_s_sum_all_starts": sum(r["solve_time_s"] for r in run_results),
            "cumulative_compile_time_s": cumulative_compile,
            "cumulative_solve_time_s": cumulative_solve,
            "cumulative_total_time_s": cumulative_compile + cumulative_solve,
            "n_starts": len(starts),
            "success": mode_best["success"],
            "nit": mode_best["nit"],
            "nfev": mode_best["nfev"],
            "message": mode_best["message"],
            "S_OB_ref": refs["S_OB"],
            "S_FB_ref": refs["S_FB"],
            "S_CT_ref": "" if refs["S_CT"] is None else refs["S_CT"],
            "rel_to_OB_percent": rel_percent(best_action, refs["S_OB"]),
            "rel_to_FB_percent": rel_percent(best_action, refs["S_FB"]),
            "rel_to_CT_percent": "" if refs["S_CT"] is None else rel_percent(best_action, refs["S_CT"]),
            "t_OB_total": refs["t_setup_OB"] + refs["t_sol_OB"],
            "t_OB_setup": refs["t_setup_OB"],
            "t_OB_sol": refs["t_sol_OB"],
            "t_FB": refs["t_FB"],
            "t_CT": "" if refs["t_CT"] is None else refs["t_CT"],
            "Vfalse": case.Vfalse,
            "Vtrue": case.Vtrue,
            "DeltaV": case.DeltaV,
        }
        history_rows.append(row)

        rc_txt = "NA" if rel_change_prev == "" else f"{100*rel_change_prev:.4f}%"
        print(f"  MODE BEST: S={mode_action:.10g}, start={mode_best['start_label']}, rel_change_prev_mode={rc_txt}, compile_once={compile_time:.4f}s")
        print(f"  GLOBAL BEST: S={best_action:.10g}, n_modes={best_modes}, rel_FB={rel_percent(best_action, refs['S_FB']):.4f}%, no_improve_count={no_improve_count}")

        previous_coeffs = mode_coeffs
        previous_modes = n_modes
        previous_mode_action = mode_action

        if no_improve_count >= PATIENCE:
            converged = True
            stop_reason = f"converged: global best action did not improve by > {TOLERANCE} for {PATIENCE} consecutive mode steps"
            print(f"STOP: {stop_reason}")
            break

    refs = case.refs
    summary = {
        "nphi": case.nphi,
        "delta": case.delta,
        "action_dim": ACTION_DIM,
        "converged": converged,
        "stop_reason": stop_reason,
        "best_n_modes": best_modes,
        "S_ours": best_action,
        "S_OB_ref": refs["S_OB"],
        "S_FB_ref": refs["S_FB"],
        "S_CT_ref": "" if refs["S_CT"] is None else refs["S_CT"],
        "rel_to_OB_percent": rel_percent(best_action, refs["S_OB"]),
        "rel_to_FB_percent": rel_percent(best_action, refs["S_FB"]),
        "rel_to_CT_percent": "" if refs["S_CT"] is None else rel_percent(best_action, refs["S_CT"]),
        "t_ours_solve": cumulative_solve,
        "t_ours_total": cumulative_compile + cumulative_solve,
        "t_ours_compile": cumulative_compile,
        "t_OB_total": refs["t_setup_OB"] + refs["t_sol_OB"],
        "t_OB_setup": refs["t_setup_OB"],
        "t_OB_sol": refs["t_sol_OB"],
        "t_FB": refs["t_FB"],
        "t_CT": "" if refs["t_CT"] is None else refs["t_CT"],
        "last_n_modes_checked": history_rows[-1]["n_modes"] if history_rows else "",
        "n_modes_checked": len(history_rows),
        "n_grid": N_GRID,
        "tolerance": TOLERANCE,
        "patience": PATIENCE,
        "n_start": N_START,
        "n_max": N_MAX,
        "n_step": N_STEP,
        "n_random_starts": N_RANDOM_STARTS,
        "bound_factor": BOUND_FACTOR,
        "Vfalse": case.Vfalse,
        "Vtrue": case.Vtrue,
        "DeltaV": case.DeltaV,
    }

    return history_rows, summary



def _as_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def make_comparison_plots(df_sum, plot_dir=PLOT_DIR):
    """
    Make action and timing comparison plots.

    Curves:
      - Our JAX-Fourier result
      - published OptiBounce
      - published FindBounce
      - published CosmoTransitions where available

    Outputs:
      plots/optibounce_table_D3_action_vs_nphi.png/pdf
      plots/optibounce_table_D3_time_vs_nphi.png/pdf
    """
    outdir = Path(plot_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = df_sum.copy()
    df["nphi"] = _as_numeric(df["nphi"])
    numeric_cols = [
        "S_ours", "S_OB_ref", "S_FB_ref", "S_CT_ref",
        "t_ours_total", "t_OB_total", "t_FB", "t_CT",
        "rel_to_OB_percent", "rel_to_FB_percent", "rel_to_CT_percent",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _as_numeric(df[col])

    df = df.sort_values("nphi")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # -------------------------
    # Action comparison
    # -------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    ax.plot(df["nphi"], df["S_ours"], marker="o", label="JAX-Fourier")
    ax.plot(df["nphi"], df["S_OB_ref"], marker="s", linestyle="--", label="OptiBounce")
    ax.plot(df["nphi"], df["S_FB_ref"], marker="^", linestyle="--", label="FindBounce")

    if "S_CT_ref" in df.columns and df["S_CT_ref"].notna().any():
        ct = df[df["S_CT_ref"].notna()]
        ax.plot(ct["nphi"], ct["S_CT_ref"], marker="D", linestyle="--", label="CosmoTransitions")

    ax.set_xlabel(r"Number of fields $N_\phi$")
    ax.set_ylabel(r"Action $S_3$")
    ax.set_title(r"OptiBounce Table-I benchmark: action comparison")
    ax.grid(True, alpha=0.30)
    ax.legend(frameon=True)
    ax.set_xticks(list(range(int(df["nphi"].min()), int(df["nphi"].max()) + 1, 1)))

    fig.tight_layout()
    png = outdir / "optibounce_table_D3_action_vs_nphi.png"
    pdf = outdir / "optibounce_table_D3_action_vs_nphi.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print("Saved", png)
    print("Saved", pdf)

    # -------------------------
    # Time comparison
    # -------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    ax.plot(df["nphi"], df["t_ours_total"], marker="o", label="JAX-Fourier total")
    ax.plot(df["nphi"], df["t_OB_total"], marker="s", linestyle="--", label="OptiBounce total")
    ax.plot(df["nphi"], df["t_FB"], marker="^", linestyle="--", label="FindBounce")

    if "t_CT" in df.columns and df["t_CT"].notna().any():
        ct = df[df["t_CT"].notna()]
        ax.plot(ct["nphi"], ct["t_CT"], marker="D", linestyle="--", label="CosmoTransitions")

    ax.set_xlabel(r"Number of fields $N_\phi$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title(r"OptiBounce Table-I benchmark: runtime comparison")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(frameon=True)
    ax.set_xticks(list(range(int(df["nphi"].min()), int(df["nphi"].max()) + 1, 1)))

    fig.tight_layout()
    png = outdir / "optibounce_table_D3_time_vs_nphi.png"
    pdf = outdir / "optibounce_table_D3_time_vs_nphi.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print("Saved", png)
    print("Saved", pdf)

    # -------------------------
    # Optional percent-deviation plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(df["nphi"], df["rel_to_OB_percent"], marker="o", label="vs OptiBounce")
    ax.plot(df["nphi"], df["rel_to_FB_percent"], marker="s", label="vs FindBounce")
    if "rel_to_CT_percent" in df.columns and df["rel_to_CT_percent"].notna().any():
        ct = df[df["rel_to_CT_percent"].notna()]
        ax.plot(ct["nphi"], ct["rel_to_CT_percent"], marker="D", label="vs CosmoTransitions")
    ax.set_xlabel(r"Number of fields $N_\phi$")
    ax.set_ylabel("Relative action difference [%]")
    ax.set_title(r"JAX-Fourier action deviation from published references")
    ax.grid(True, alpha=0.30)
    ax.legend(frameon=True)
    ax.set_xticks(list(range(int(df["nphi"].min()), int(df["nphi"].max()) + 1, 1)))
    fig.tight_layout()
    png = outdir / "optibounce_table_D3_action_deviation_vs_nphi.png"
    pdf = outdir / "optibounce_table_D3_action_deviation_vs_nphi.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print("Saved", png)
    print("Saved", pdf)



def main():
    all_history = []
    summaries = []

    print("\nAdaptive JAX-Fourier OptiBounce/Table-I benchmark")
    print("=" * 100)
    print(f"ACTION_DIM        = {ACTION_DIM}")
    print(f"NPHI_VALUES       = {NPHI_VALUES}")
    print(f"N_START,MAX,STEP  = {N_START}, {N_MAX}, {N_STEP}")
    print(f"TOLERANCE         = {TOLERANCE}")
    print(f"PATIENCE          = {PATIENCE}")
    print(f"N_GRID            = {N_GRID}")
    print(f"MAXITER           = {MAXITER}")
    print("References are the published Dimension-3/Table-I values.")
    print("Stopping does NOT use reference actions.")
    print("No special global rescue is applied to any individual N_phi.")
    print()

    for nphi in NPHI_VALUES:
        print("\n" + "#" * 100)
        print(f"nphi = {nphi}")

        t0 = time.perf_counter()
        case = make_case(nphi)
        preprocess_time = time.perf_counter() - t0

        print(f"preprocess minima time = {preprocess_time:.6f} s")
        print(f"Vfalse = {case.Vfalse:.12g}")
        print(f"Vtrue  = {case.Vtrue:.12g}")
        print(f"DeltaV = {case.DeltaV:.12g}")
        print(f"Published FB reference = {case.refs['S_FB']}")

        hist, summ = adaptive_scan_case(case)

        for row in hist:
            row["preprocess_minima_time_s"] = preprocess_time
        summ["preprocess_minima_time_s"] = preprocess_time

        all_history.extend(hist)
        summaries.append(summ)

    df_hist = pd.DataFrame(all_history)
    df_sum = pd.DataFrame(summaries)

    df_hist.to_csv(HISTORY_CSV, index=False)
    df_sum.to_csv(SUMMARY_CSV, index=False)

    print("\n\nFINAL SUMMARY")
    print("=" * 170)
    cols = [
        "nphi",
        "converged",
        "best_n_modes",
        "S_ours",
        "S_OB_ref",
        "S_FB_ref",
        "S_CT_ref",
        "rel_to_FB_percent",
        "t_ours_total",
        "t_OB_total",
        "t_FB",
        "t_CT",
        "last_n_modes_checked",
        "stop_reason",
    ]
    print(df_sum[cols].to_string(index=False))

    print(f"\nSaved history: {HISTORY_CSV}")
    print(f"Saved summary: {SUMMARY_CSV}")
    make_comparison_plots(df_sum, PLOT_DIR)
    print(f"Saved plots in: {PLOT_DIR}/")


if __name__ == "__main__":
    main()
