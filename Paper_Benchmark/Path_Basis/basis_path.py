#!/usr/bin/env python3
"""
Hybrid-basis benchmark: Fourier vs wavelet/local vs hybrid.

This script compares pure Fourier, pure local/wavelet, and hybrid Fourier+local bases.

It includes two benchmark types:

1. Target fitting benchmark:
   No potential, no bounce action.
   Tests representation power directly by fitting known target paths.

2. Bounce-action benchmark:
   Uses the same O(4) Espinosa reduced action used in your Fourier code.
   Potentials:
       A. optibounce_table, n_phi = 3...10
       B. kinked_valley_2d, widths = 0.25, 0.15, 0.08

Outputs:
    hybrid_target_fit_results.csv
    hybrid_bounce_results.csv
    plots_hybrid_target_fit/*.png

Install:
    python3 -m pip install numpy scipy PyWavelets matplotlib

Run:
    python3 hybrid_basis_benchmark.py
"""

import csv
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

try:
    import pywt
    HAS_PYWT = True
except Exception:
    pywt = None
    HAS_PYWT = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False


# ============================================================
# User settings
# ============================================================

RUN_TARGET_FIT = True
RUN_BOUNCE = False
MAKE_PLOTS = True

TOTAL_BASIS_VALUES = [5, 8, 10, 15]
HYBRID_FOURIER_FRACTION = 0.6

OPTIBOUNCE_NPHI_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
KINK_WIDTH_VALUES = [0.25, 0.15, 0.08]

N_POINTS_TARGET = 1200
N_POINTS_BOUNCE = 180

OPTIMIZER = "powell"
MAXITER_BOUNCE = 450
REPEAT_BOUNCE = 1

BASIS_FAMILIES = [
    "fourier",
    "gaussian_wide",
    "waveletwide_gaus1",
    "multiscale_mexh",
    "multiscale_gaus1",
    "hybrid_fourier_gaussian_wide",
    "hybrid_fourier_waveletwide_gaus1",
    "hybrid_fourier_multiscale_mexh",
    "hybrid_fourier_multiscale_gaus1",
]


OPTIBOUNCE_DATA = {
    3:  {"delta": 0.065, "c": [0.684373, 0.181928, 0.295089], "FBrefD4": 7928.0},
    4:  {"delta": 0.11,  "c": [0.534808, 0.77023, 0.838912, 0.00517238], "FBrefD4": 5129.18},
    5:  {"delta": 0.13,  "c": [0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "FBrefD4": 5111.98},
    6:  {"delta": 0.15,  "c": [0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "FBrefD4": 5577.5},
    7:  {"delta": 0.2,   "c": [0.5233, 0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.517238], "FBrefD4": 4339.15},
    8:  {"delta": 0.22,  "c": [0.2434, 0.5233, 0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.51723], "FBrefD4": 4418.09},
    9:  {"delta": 0.29,  "c": [0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "FBrefD4": 2708.05},
    10: {"delta": 0.27,  "c": [0.12, 0.21, 0.24, 0.52, 0.34, 0.47, 0.23, 0.57, 0.14, 0.52], "FBrefD4": 3705.49},
}


@dataclass
class BasisData:
    basis_family: str
    n_total: int
    n_fourier: int
    n_local: int
    t_grid: np.ndarray
    B: np.ndarray
    labels: list


@dataclass
class PotentialCase:
    case_name: str
    label: str
    nphi: int
    false_vac: np.ndarray
    true_vac: np.ndarray
    Vfalse: float
    Vtrue: float
    reference_action: Optional[float]
    potential: Callable[[np.ndarray], np.ndarray]


# ============================================================
# Target paths
# ============================================================

def target_smooth_sine(t):
    return 0.5 * np.sin(np.pi * t)


def target_narrow_bump(t):
    return 0.8 * np.exp(-0.5 * ((t - 0.55) / 0.045)**2)


def target_double_bump(t):
    return 0.7 * np.exp(-0.5 * ((t - 0.32) / 0.055)**2) - 0.45 * np.exp(-0.5 * ((t - 0.72) / 0.075)**2)


def target_tanh_wall_sharp(t):
    return 0.6 * np.tanh((t - 0.52) / 0.035)


def target_abs_cusp(t):
    return 0.8 * np.abs(t - 0.5)


def target_step_like(t):
    return 0.7 / (1.0 + np.exp(-(t - 0.5) / 0.015))


TARGETS = {
    "smooth_sine": target_smooth_sine,
    "narrow_bump": target_narrow_bump,
    "double_bump": target_double_bump,
    "tanh_wall_sharp": target_tanh_wall_sharp,
    "abs_cusp": target_abs_cusp,
    "step_like": target_step_like,
}


# ============================================================
# Potentials
# ============================================================

def optibounce_potential(phi, c, delta):
    phi = np.asarray(phi, dtype=float)
    c = np.asarray(c, dtype=float)
    A = np.sum(c * (phi - 1.0)**2, axis=-1) - delta
    B = np.sum(phi**2, axis=-1)
    return A * B


def optibounce_gradient(phi, c, delta):
    phi = np.asarray(phi, dtype=float)
    c = np.asarray(c, dtype=float)
    A = np.sum(c * (phi - 1.0)**2, axis=-1) - delta
    B = np.sum(phi**2, axis=-1)
    return 2.0 * c * (phi - 1.0) * np.expand_dims(B, axis=-1) + 2.0 * phi * np.expand_dims(A, axis=-1)


def make_optibounce_case(nphi):
    data = OPTIBOUNCE_DATA[nphi]
    c = np.array(data["c"], dtype=float)
    delta = float(data["delta"])
    false_vac = np.zeros(nphi, dtype=float)

    def obj(x):
        return float(optibounce_potential(x, c, delta))

    def jac(x):
        return optibounce_gradient(x, c, delta)

    res = minimize(obj, np.ones(nphi), jac=jac, method="BFGS", options={"gtol": 1e-10, "maxiter": 5000})
    true_vac = res.x.astype(float)

    def V(phi):
        return optibounce_potential(phi, c, delta)

    Vfalse = float(V(false_vac))
    Vtrue = float(V(true_vac))

    if Vtrue >= Vfalse:
        raise RuntimeError(f"Bad vacuum ordering for optibounce nphi={nphi}")

    return PotentialCase(
        case_name="optibounce_table",
        label=f"optibounce_n{nphi}",
        nphi=nphi,
        false_vac=false_vac,
        true_vac=true_vac,
        Vfalse=Vfalse,
        Vtrue=Vtrue,
        reference_action=float(data["FBrefD4"]),
        potential=V,
    )


def make_kinked_valley_case(width, eps=0.08, A=0.85, my=6.0):
    def y0(x):
        return A * np.tanh(x / width)

    def V_single(z):
        x, y = float(z[0]), float(z[1])
        return 0.25*(x*x - 1.0)**2 - eps*x + 0.5*my*my*(y - y0(x))**2

    def V(phi):
        phi = np.asarray(phi, dtype=float)
        x = phi[..., 0]
        y = phi[..., 1]
        return 0.25*(x*x - 1.0)**2 - eps*x + 0.5*my*my*(y - y0(x))**2

    starts = [
        np.array([-1.0, y0(-1.0)]),
        np.array([+1.0, y0(+1.0)]),
        np.array([-0.7, y0(-0.7)]),
        np.array([+0.7, y0(+0.7)]),
    ]

    minima = []
    for st in starts:
        res = minimize(V_single, st, method="BFGS", options={"gtol": 1e-11, "maxiter": 5000})
        x = res.x.astype(float)
        val = float(V_single(x))
        if all(np.linalg.norm(x - old[0]) > 1e-7 for old in minima):
            minima.append((x, val))

    if len(minima) < 2:
        raise RuntimeError(f"Could not find two minima for kink width={width}")

    minima = sorted(minima, key=lambda p: p[1])
    true_vac = minima[0][0]
    false_vac = minima[-1][0]

    Vtrue = float(V_single(true_vac))
    Vfalse = float(V_single(false_vac))

    if Vtrue >= Vfalse:
        raise RuntimeError(f"Bad vacuum ordering for kink width={width}")

    return PotentialCase(
        case_name="kinked_valley_2d",
        label=f"kink_w{width:g}",
        nphi=2,
        false_vac=false_vac,
        true_vac=true_vac,
        Vfalse=Vfalse,
        Vtrue=Vtrue,
        reference_action=None,
        potential=V,
    )


# ============================================================
# Basis construction
# ============================================================

def normalize_basis_rows(B, t):
    out = np.array(B, dtype=float, copy=True)
    if out.size == 0:
        return out.reshape(0, len(t))
    for i in range(out.shape[0]):
        norm = np.sqrt(np.trapezoid(out[i]**2, t))
        if norm > 0:
            out[i] /= norm
    return out


def fourier_rows(t, n):
    if n <= 0:
        return np.zeros((0, len(t))), []
    rows = [np.sin(k*np.pi*t) for k in range(1, n + 1)]
    labels = [f"sin{k}" for k in range(1, n + 1)]
    return normalize_basis_rows(np.array(rows), t), labels


def gaussian_wide_rows(t, n):
    if n <= 0:
        return np.zeros((0, len(t))), []
    centers = np.linspace(0.04, 0.96, n)
    sigma = 0.75 / max(n, 2)
    env = np.sin(np.pi*t)
    rows = [env * np.exp(-0.5*((t-c)/sigma)**2) for c in centers]
    labels = [f"gwide{j}" for j in range(n)]
    return normalize_basis_rows(np.array(rows), t), labels


def continuous_wavelet_values(wavelet_name, x):
    if not HAS_PYWT:
        raise ImportError("PyWavelets not installed. Run: python3 -m pip install PyWavelets")
    wavelet = pywt.ContinuousWavelet(wavelet_name)
    psi, x_psi = wavelet.wavefun(level=10)
    return np.interp(x, x_psi, psi, left=0.0, right=0.0)


def waveletwide_rows(t, n, wavelet_name):
    if n <= 0:
        return np.zeros((0, len(t))), []
    centers = np.linspace(0.04, 0.96, n)
    scale = 0.85 / max(n, 2)
    env = np.sin(np.pi*t)
    rows = []
    labels = []
    for j, c in enumerate(centers):
        psi = continuous_wavelet_values(wavelet_name, (t-c)/scale)
        rows.append(env * psi)
        labels.append(f"ww_{wavelet_name}_{j}")
    return normalize_basis_rows(np.array(rows), t), labels


def multiscale_wavelet_rows(t, n_total, wavelet_name, scale_factors=(0.12, 0.25, 0.50, 0.90)):
    if n_total <= 0:
        return np.zeros((0, len(t))), []
    if not HAS_PYWT:
        raise ImportError("PyWavelets not installed. Run: python3 -m pip install PyWavelets")

    n_scales = len(scale_factors)
    counts = [n_total // n_scales] * n_scales
    for i in range(n_total % n_scales):
        counts[i] += 1

    env = np.sin(np.pi*t)
    rows = []
    labels = []

    for sf, count in zip(scale_factors, counts):
        if count <= 0:
            continue
        centers = np.linspace(0.04, 0.96, count)
        scale = sf / max(count, 2)
        for j, c in enumerate(centers):
            psi = continuous_wavelet_values(wavelet_name, (t-c)/scale)
            rows.append(env * psi)
            labels.append(f"multi_{wavelet_name}_sf{sf}_{j}")

    return normalize_basis_rows(np.array(rows), t), labels


def local_rows_for_family(t, n_local, local_family):
    if local_family == "gaussian_wide":
        return gaussian_wide_rows(t, n_local)
    if local_family == "waveletwide_gaus1":
        return waveletwide_rows(t, n_local, "gaus1")
    if local_family == "multiscale_mexh":
        return multiscale_wavelet_rows(t, n_local, "mexh")
    if local_family == "multiscale_gaus1":
        return multiscale_wavelet_rows(t, n_local, "gaus1")
    raise ValueError(f"Unknown local family: {local_family}")


def split_hybrid_counts(n_total, fraction):
    n_fourier = int(math.ceil(fraction * n_total))
    n_fourier = max(1, min(n_fourier, n_total - 1))
    n_local = n_total - n_fourier
    return n_fourier, n_local


def make_basis(t, n_total, basis_family):
    basis_family = basis_family.lower()

    if basis_family == "fourier":
        B, labels = fourier_rows(t, n_total)
        return BasisData(basis_family, n_total, n_total, 0, t, B, labels)

    if basis_family in ["gaussian_wide", "waveletwide_gaus1", "multiscale_mexh", "multiscale_gaus1"]:
        B, labels = local_rows_for_family(t, n_total, basis_family)
        return BasisData(basis_family, B.shape[0], 0, B.shape[0], t, B, labels)

    if basis_family.startswith("hybrid_fourier_"):
        local_family = basis_family.replace("hybrid_fourier_", "")
        n_fourier, n_local = split_hybrid_counts(n_total, HYBRID_FOURIER_FRACTION)

        BF, labelsF = fourier_rows(t, n_fourier)
        BL, labelsL = local_rows_for_family(t, n_local, local_family)

        B = np.vstack([BF, BL])
        labels = [f"F:{x}" for x in labelsF] + [f"L:{x}" for x in labelsL]

        return BasisData(basis_family, B.shape[0], n_fourier, n_local, t, B, labels)

    raise ValueError(f"Unknown basis family: {basis_family}")


# ============================================================
# Target fitting benchmark
# ============================================================

def straight_baseline(t, y):
    return y[0] + t*(y[-1] - y[0])


def fit_basis_least_squares(t, deformation, B):
    A = B.T
    t0 = time.perf_counter()
    coeffs, residuals, rank, svals = np.linalg.lstsq(A, deformation, rcond=None)
    t1 = time.perf_counter()
    fit_def = A @ coeffs
    return coeffs, fit_def, t1-t0, rank, svals


def target_errors(t, y_target, y_fit):
    diff = y_fit - y_target
    l2_abs = np.sqrt(np.trapezoid(diff**2, t))
    l2_target = np.sqrt(np.trapezoid(y_target**2, t))
    l2_rel = l2_abs / max(l2_target, 1e-300)
    linf_abs = float(np.max(np.abs(diff)))
    amp = float(np.max(y_target) - np.min(y_target))
    linf_rel = linf_abs / max(amp, 1e-300)
    return l2_abs, l2_rel, linf_abs, linf_rel


def run_target_fit_benchmark():
    print("\n" + "#"*120)
    print("TARGET-FITTING BENCHMARK")
    print("#"*120)

    rows = []
    t = np.linspace(0.0, 1.0, N_POINTS_TARGET)

    plot_dir = Path("results")
    if MAKE_PLOTS and HAS_MPL:
        plot_dir.mkdir(exist_ok=True)

    for target_name, target_func in TARGETS.items():
        print("\n" + "="*100)
        print(f"TARGET = {target_name}")

        y_target = target_func(t)
        y0 = straight_baseline(t, y_target)
        deformation = y_target - y0

        for n_total in TOTAL_BASIS_VALUES:
            fits_for_plot = []
            print("\n" + "-"*80)
            print(f"n_total = {n_total}")

            for basis_family in BASIS_FAMILIES:
                try:
                    basis = make_basis(t, n_total, basis_family)
                except Exception as err:
                    print(f"{basis_family:>35} failed: {type(err).__name__}: {err}")
                    continue

                coeffs, fit_def, fit_time, rank, svals = fit_basis_least_squares(t, deformation, basis.B)
                y_fit = y0 + fit_def
                l2_abs, l2_rel, linf_abs, linf_rel = target_errors(t, y_target, y_fit)

                row = {
                    "benchmark": "target_fit",
                    "target": target_name,
                    "basis_family": basis_family,
                    "n_total": basis.n_total,
                    "n_fourier": basis.n_fourier,
                    "n_local": basis.n_local,
                    "fit_time_s": fit_time,
                    "rank": int(rank),
                    "l2_abs": l2_abs,
                    "l2_rel": l2_rel,
                    "linf_abs": linf_abs,
                    "linf_rel": linf_rel,
                    "max_coeff_abs": float(np.max(np.abs(coeffs))) if len(coeffs) else 0.0,
                    "condition_est": float(svals[0]/svals[-1]) if len(svals) and svals[-1] > 0 else np.inf,
                }
                rows.append(row)

                print(
                    f"{basis_family:>35} "
                    f"F={basis.n_fourier:2d} L={basis.n_local:2d} "
                    f"L2rel={l2_rel:10.3e} "
                    f"LinfRel={linf_rel:10.3e}"
                )

                if MAKE_PLOTS and HAS_MPL and n_total in [5, 10, 15]:
                    if basis_family in [
                        "fourier",
                        "gaussian_wide",
                        "multiscale_mexh",
                        "hybrid_fourier_gaussian_wide",
                        "hybrid_fourier_multiscale_mexh",
                    ]:
                        fits_for_plot.append((basis_family, y_fit))

            if MAKE_PLOTS and HAS_MPL and fits_for_plot:
                plt.figure(figsize=(8, 5))
                plt.plot(t, y_target, label="target", linewidth=2.5)
                plt.plot(t, y0, "--", label="straight", linewidth=1.4)
                for label, y_fit in fits_for_plot:
                    plt.plot(t, y_fit, label=label, linewidth=1.0)
                plt.title(f"{target_name}, n_total={n_total}")
                plt.xlabel("t")
                plt.ylabel("y(t)")
                plt.legend(fontsize=7)
                plt.tight_layout()
                out = plot_dir / f"{target_name}_n{n_total}.png"
                plt.savefig(out, dpi=160)
                plt.close()

    best_by_key = {}
    for r in rows:
        key = (r["target"], r["n_total"])
        best_by_key[key] = min(best_by_key.get(key, r["l2_rel"]), r["l2_rel"])

    for r in rows:
        key = (r["target"], r["n_total"])
        best = best_by_key[key]
        r["best_l2_rel_for_target_ntotal"] = best
        r["rel_to_best_l2"] = r["l2_rel"] / max(best, 1e-300)

    save_csv(rows, "hybrid_target_fit_results.csv")

    print("\nTARGET-FIT SUMMARY: best basis by target and n_total")
    print("="*110)
    print(f"{'target':>20} {'n_total':>8} {'best_basis':>35} {'best_L2rel':>14}")
    print("-"*110)
    for target_name in TARGETS:
        for n_total in TOTAL_BASIS_VALUES:
            subset = [r for r in rows if r["target"] == target_name and r["n_total"] == n_total]
            if not subset:
                continue
            best = min(subset, key=lambda r: r["l2_rel"])
            print(f"{target_name:>20} {n_total:8d} {best['basis_family']:>35} {best['l2_rel']:14.3e}")

    return rows


# ============================================================
# Bounce benchmark
# ============================================================

def Vt_smooth(t, case):
    return case.Vfalse + t*t*(3.0 - 2.0*t)*(case.Vtrue - case.Vfalse)


def base_path(case, t):
    return case.false_vac[None, :] + t[:, None]*(case.true_vac - case.false_vac)[None, :]


def deformed_path(coeffs, case, basis):
    C = np.asarray(coeffs, dtype=float).reshape(case.nphi, basis.n_total)
    return base_path(case, basis.t_grid) + basis.B.T @ C.T


def reduced_action_O4(coeffs, case, basis, penalty=1e100):
    t = basis.t_grid
    phi = deformed_path(coeffs, case, basis)
    Vphi = case.potential(phi)
    Vt = Vt_smooth(t, case)

    if not np.all(np.isfinite(Vphi)):
        return penalty

    total = 0.0
    for i in range(len(t)-1):
        dVt = Vt[i+1] - Vt[i]
        den = -(dVt**3)
        if den <= 1e-300 or not np.isfinite(den):
            return penalty
        Vbar_minus_Vtbar = Vphi[i] + Vphi[i+1] - Vt[i] - Vt[i+1]
        ds = np.linalg.norm(phi[i+1] - phi[i])
        term = (Vbar_minus_Vtbar**2) * (ds**4) / den
        if not np.isfinite(term):
            return penalty
        total += term

    return float(27.0*np.pi**2*total/2.0)


def optimize_bounce_case(case, basis, repeat_seed=0):
    nvar = case.nphi * basis.n_total
    path_length = np.linalg.norm(case.true_vac - case.false_vac)
    amp_bound = path_length / (2.0*np.sqrt(max(basis.n_total, 1)))
    bounds = [(-amp_bound, amp_bound)] * nvar

    seed = 10007 + repeat_seed + 101*case.nphi + 17*basis.n_total
    rng = np.random.default_rng(seed)

    if repeat_seed == 0:
        x0 = np.zeros(nvar)
    else:
        x0 = rng.normal(scale=0.03*amp_bound, size=nvar)
        x0 = np.clip(x0, -amp_bound, amp_bound)

    def obj(x):
        return reduced_action_O4(x, case, basis)

    t0 = time.perf_counter()

    if OPTIMIZER.lower() == "powell":
        res = minimize(
            obj,
            x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": MAXITER_BOUNCE, "xtol": 1e-4, "ftol": 1e-7, "disp": False},
        )
    elif OPTIMIZER.lower() == "lbfgsb":
        res = minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAXITER_BOUNCE, "ftol": 1e-10},
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    t1 = time.perf_counter()

    return {
        "S4": float(res.fun),
        "time_s": t1-t0,
        "success": bool(res.success),
        "nfev": int(getattr(res, "nfev", -1)),
        "message": str(res.message),
    }


def run_bounce_benchmark():
    print("\n" + "#"*120)
    print("BOUNCE-ACTION BENCHMARK")
    print("#"*120)

    rows = []
    t = np.linspace(0.0, 1.0, N_POINTS_BOUNCE)

    cases = []
    for nphi in OPTIBOUNCE_NPHI_VALUES:
        cases.append(make_optibounce_case(nphi))
    for w in KINK_WIDTH_VALUES:
        cases.append(make_kinked_valley_case(w))

    for case in cases:
        print("\n" + "="*120)
        print(f"CASE = {case.label} ({case.case_name}), nphi={case.nphi}")
        print(f"Vfalse = {case.Vfalse:.12g}")
        print(f"Vtrue  = {case.Vtrue:.12g}")
        print(f"DeltaV = {case.Vfalse - case.Vtrue:.12g}")
        print(f"reference = {case.reference_action}")

        for n_total in TOTAL_BASIS_VALUES:
            print("\n" + "-"*100)
            print(f"n_total = {n_total}")

            for basis_family in BASIS_FAMILIES:
                try:
                    basis = make_basis(t, n_total, basis_family)
                except Exception as err:
                    print(f"{basis_family:>35} failed: {type(err).__name__}: {err}")
                    continue

                actions = []
                times = []
                successes = []
                nfevs = []

                for rep in range(REPEAT_BOUNCE):
                    out = optimize_bounce_case(case, basis, repeat_seed=rep)
                    actions.append(out["S4"])
                    times.append(out["time_s"])
                    successes.append(out["success"])
                    nfevs.append(out["nfev"])

                S_med = float(np.median(actions))
                t_med = float(np.median(times))

                rel_ref = ""
                if case.reference_action is not None:
                    rel_ref = abs(S_med - case.reference_action) / case.reference_action

                row = {
                    "benchmark": "bounce_action",
                    "case_name": case.case_name,
                    "label": case.label,
                    "nphi": case.nphi,
                    "basis_family": basis_family,
                    "n_total": basis.n_total,
                    "n_fourier": basis.n_fourier,
                    "n_local": basis.n_local,
                    "repeat": REPEAT_BOUNCE,
                    "S4_med": S_med,
                    "S4_min": float(np.min(actions)),
                    "S4_max": float(np.max(actions)),
                    "time_med_s": t_med,
                    "time_min_s": float(np.min(times)),
                    "time_max_s": float(np.max(times)),
                    "success_count": int(sum(successes)),
                    "nfev_med": int(np.median(nfevs)),
                    "reference_action": "" if case.reference_action is None else case.reference_action,
                    "rel_to_reference": rel_ref,
                    "Vfalse": case.Vfalse,
                    "Vtrue": case.Vtrue,
                    "DeltaV": case.Vfalse - case.Vtrue,
                }
                rows.append(row)

                reltxt = "NA" if rel_ref == "" else f"{100*rel_ref:.4f}%"
                print(
                    f"{basis_family:>35} "
                    f"F={basis.n_fourier:2d} L={basis.n_local:2d} "
                    f"S4={S_med:12.6g} "
                    f"time={t_med:8.3f}s "
                    f"rel_ref={reltxt}"
                )

    best_by_key = {}
    for r in rows:
        key = (r["label"], r["n_total"])
        best_by_key[key] = min(best_by_key.get(key, r["S4_med"]), r["S4_med"])

    for r in rows:
        key = (r["label"], r["n_total"])
        best = best_by_key[key]
        r["best_S4_for_label_ntotal"] = best
        r["rel_to_best_scan_same_ntotal"] = abs(r["S4_med"] - best) / best

    save_csv(rows, "hybrid_bounce_results.csv")

    print("\nBOUNCE SUMMARY: best basis by case and n_total")
    print("="*120)
    print(f"{'label':>16} {'n_total':>8} {'best_basis':>35} {'best_S4':>14} {'rel_ref/best':>14}")
    print("-"*120)

    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])

    for label in labels:
        for n_total in TOTAL_BASIS_VALUES:
            subset = [r for r in rows if r["label"] == label and r["n_total"] == n_total]
            if not subset:
                continue
            best = min(subset, key=lambda r: r["S4_med"])
            if best["reference_action"] != "":
                metric = abs(best["S4_med"] - best["reference_action"]) / best["reference_action"]
                metric_text = f"{100*metric:.4f}% ref"
            else:
                metric_text = "0.0000% best"
            print(
                f"{label:>16} {n_total:8d} {best['basis_family']:>35} "
                f"{best['S4_med']:14.6g} {metric_text:>14}"
            )

    return rows


def save_csv(rows, filename):
    if not rows:
        print(f"No rows to save for {filename}.")
        return
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {filename}")


def main():
    print("\nHybrid-basis benchmark")
    print("="*120)
    print(f"PyWavelets available : {HAS_PYWT}")
    print(f"matplotlib available : {HAS_MPL}")
    print(f"RUN_TARGET_FIT       : {RUN_TARGET_FIT}")
    print(f"RUN_BOUNCE           : {RUN_BOUNCE}")
    print(f"TOTAL_BASIS_VALUES   : {TOTAL_BASIS_VALUES}")
    print(f"Hybrid Fourier frac  : {HYBRID_FOURIER_FRACTION}")

    if not HAS_PYWT:
        print("\nWARNING: PyWavelets missing. Install with:")
        print("    python3 -m pip install PyWavelets\n")

    if RUN_TARGET_FIT:
        run_target_fit_benchmark()

    if RUN_BOUNCE:
        run_bounce_benchmark()

    print("\nDone.")
    print("Main outputs:")
    print("  hybrid_target_fit_results.csv")
    print("  hybrid_bounce_results.csv")
    if MAKE_PLOTS and HAS_MPL:
        print("  plots_hybrid_target_fit/")


if __name__ == "__main__":
    main()
