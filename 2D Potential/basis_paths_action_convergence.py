#!/usr/bin/env python3
"""
paper_style_optibounce2d_basis_paths.py

Standalone paper-style figure generator for a genuine 2D OptiBounce-like potential.

This script makes one clean contour/path plot per basis, so the figure does not
become overcrowded. It is meant for visual explanation: good bases show smooth
endpoint-safe path deformation, while bad bases can be shown separately as
diagnostic examples.

Potential
---------
We use the 2D version of the OptiBounce benchmark form

    V(phi) = [ c1 (phi1-1)^2 + c2 (phi2-1)^2 - delta ] (phi1^2 + phi2^2)

The origin is an exact stationary false vacuum because of the overall |phi|^2
factor. The true vacuum is found by minimizing V from several initial guesses.

The default parameters use the first two coefficients from the published
OptiBounce n_phi=3 benchmark, together with its delta:

    c = (0.684373, 0.181928), delta = 0.065

This is not the published table case itself, which starts at n_phi=3. It is a
2D visualization case using the same potential family.

Outputs
-------
plots/optibounce2d/main_body/
    optibounce2d_fourier_path_contour.png
    optibounce2d_bspline_local_path_contour.png
    optibounce2d_hybrid_fourier_bspline_path_contour.png
    optibounce2d_chebyshev_path_contour.png
    optibounce2d_wavelet_db4_path_contour.png

For Fourier only, it also saves:
    optibounce2d_fourier_convergence.png
    optibounce2d_fourier_profiles.png

Run examples
------------
    python3 paper_style_optibounce2d_basis_paths.py --basis fourier --no-show
    python3 paper_style_optibounce2d_basis_paths.py --basis bspline_local --no-show
    python3 paper_style_optibounce2d_basis_paths.py --basis chebyshev --no-show
    python3 paper_style_optibounce2d_basis_paths.py --basis wavelet_db4 --no-show
    python3 paper_style_optibounce2d_basis_paths.py --basis all --no-show
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from math import gamma

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =============================================================================
# Settings
# =============================================================================

ACTION_DIM = 4

# 2D OptiBounce-like parameters.
C = np.array([0.684373, 0.181928], dtype=float)
DELTA = 0.065

# Path/action grid
N_GRID = 420

# Mode scan. These are enough for the visual comparison.
MODES_TO_STORE = [1, 2, 3, 4, 5, 7, 10]

BASIS_LIST = [
    "fourier",
    "bspline_local",
    "hybrid_fourier_bspline",
    "chebyshev",
    "wavelet_db4",
]

MAXITER = 1500

# Output
OUTDIR = Path("plots") / "optibounce2d" / "main_body"


# =============================================================================
# Plot style
# =============================================================================

def apply_style():
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
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.facecolor": "white",
        "legend.edgecolor": "#d0d0d0",
        "grid.color": "#b0b0b0",
        "grid.alpha": 0.30,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.5,
        "savefig.bbox": "tight",
        "savefig.dpi": 270,
    })


def style_axes(ax):
    ax.grid(True, which="both")
    ax.tick_params(direction="in", top=True, right=True, length=4)
    for spine in ax.spines.values():
        spine.set_color("#2b2b2b")


# =============================================================================
# Basis labels and styles
# =============================================================================

def nice_basis_label(basis_name: str) -> str:
    labels = {
        "fourier": "Fourier",
        "bspline_local": "B-spline",
        "hybrid_fourier_bspline": "Fourier+B-spline",
        "chebyshev": "Chebyshev",
        "wavelet_db4": "Wavelet db4",
    }
    return labels.get(basis_name, basis_name)


def basis_color(basis_name: str) -> str:
    colors = {
        "fourier": "#1f4aff",
        "bspline_local": "#168a2f",
        "hybrid_fourier_bspline": "#7b2cbf",
        "chebyshev": "#c7361d",
        "wavelet_db4": "#d6278b",
    }
    return colors.get(basis_name, "#1f4aff")


def basis_linestyle(basis_name: str) -> str:
    styles = {
        "fourier": "-",
        "bspline_local": "-",
        "hybrid_fourier_bspline": "-",
        "chebyshev": "--",
        "wavelet_db4": ":",
    }
    return styles.get(basis_name, "-")


# =============================================================================
# Potential and vacua
# =============================================================================

def V(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    A = np.sum(C * (phi - 1.0)**2, axis=-1) - DELTA
    R2 = np.sum(phi**2, axis=-1)
    return A * R2


def gradV(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    A = np.sum(C * (phi - 1.0)**2) - DELTA
    R2 = np.sum(phi**2)
    return 2.0 * C * (phi - 1.0) * R2 + 2.0 * phi * A


def find_true_vacuum() -> tuple[np.ndarray, float]:
    guesses = []
    for x in np.linspace(-0.4, 1.8, 8):
        for y in np.linspace(-0.4, 1.8, 8):
            guesses.append(np.array([x, y], dtype=float))

    minima = []
    for g in guesses:
        res = minimize(
            lambda z: float(V(z)),
            g,
            jac=gradV,
            method="BFGS",
            options={"gtol": 1e-12, "maxiter": 5000},
        )
        if not res.success:
            continue

        x = res.x
        val = float(V(x))
        if not any(np.linalg.norm(x - m[0]) < 1e-7 for m in minima):
            minima.append((x, val))

    if not minima:
        raise RuntimeError("Could not find any true-vacuum candidate.")

    minima = sorted(minima, key=lambda z: z[1])
    return minima[0]


def hessian_at_origin() -> np.ndarray:
    # Near origin V ~ A(0) |phi|^2, so Hessian = 2 A(0) I.
    A0 = np.sum(C) - DELTA
    return 2.0 * A0 * np.eye(2)


# =============================================================================
# Endpoint-safe bases
# =============================================================================

def normalize_rows(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    for i in range(B.shape[0]):
        m = np.max(np.abs(B[i]))
        if m > 0:
            B[i] /= m
    return B


def fourier_basis(t: np.ndarray, n_modes: int) -> np.ndarray:
    B = np.array([np.sin((k + 1) * np.pi * t) for k in range(n_modes)], dtype=float)
    return normalize_rows(B)


def chebyshev_basis(t: np.ndarray, n_modes: int) -> np.ndarray:
    x = 2.0 * t - 1.0
    env = t * (1.0 - t)
    rows = []
    for k in range(n_modes):
        Tk = np.cos((k + 1) * np.arccos(np.clip(x, -1.0, 1.0)))
        rows.append(env * Tk)
    return normalize_rows(np.array(rows, dtype=float))


def gaussian_basis(t: np.ndarray, n_modes: int) -> np.ndarray:
    env = t * (1.0 - t)
    centers = np.linspace(0.1, 0.9, n_modes)
    width = 0.75 / max(n_modes, 2)
    rows = [env * np.exp(-0.5 * ((t - c) / width) ** 2) for c in centers]
    return normalize_rows(np.array(rows, dtype=float))


def bspline_basis(t: np.ndarray, n_modes: int) -> np.ndarray:
    from scipy.interpolate import BSpline

    degree = min(3, max(0, n_modes - 1))
    n_coeff = n_modes

    if n_coeff <= degree:
        return gaussian_basis(t, n_modes)

    n_knots = n_coeff + degree + 1
    interior_count = n_knots - 2 * (degree + 1)

    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    else:
        knots = np.r_[np.zeros(degree + 1), np.ones(degree + 1)]

    env = t * (1.0 - t)
    rows = []
    for i in range(n_coeff):
        coeff = np.zeros(n_coeff)
        coeff[i] = 1.0
        spl = BSpline(knots, coeff, degree, extrapolate=False)
        vals = np.nan_to_num(spl(t), nan=0.0)
        rows.append(env * vals)

    return normalize_rows(np.array(rows, dtype=float))


def hybrid_fourier_bspline_basis(t: np.ndarray, n_modes: int) -> np.ndarray:
    nF = max(1, int(np.ceil(0.6 * n_modes)))
    nL = max(0, n_modes - nF)

    Bf = fourier_basis(t, nF)

    if nL == 0:
        return Bf

    Bl = bspline_basis(t, nL)
    return normalize_rows(np.vstack([Bf, Bl]))


def wavelet_basis(t: np.ndarray, n_modes: int, wavelet_name: str = "db4") -> np.ndarray:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("PyWavelets is needed. Install with: pip install PyWavelets") from exc

    env = t * (1.0 - t)
    wavelet = pywt.Wavelet(wavelet_name)
    wf = wavelet.wavefun(level=8)

    if len(wf) == 3:
        _, psi, xw = wf
    elif len(wf) == 5:
        _, psi, _, _, xw = wf
    else:
        raise RuntimeError(f"Unexpected wavefun output for {wavelet_name}")

    xw = np.asarray(xw, dtype=float)
    psi = np.asarray(psi, dtype=float)
    xw_unit = (xw - np.min(xw)) / (np.max(xw) - np.min(xw))

    centers = np.linspace(0.1, 0.9, n_modes)
    width = 0.8 / max(n_modes, 2)

    rows = []
    for c in centers:
        u = (t - c) / width + 0.5
        vals = np.interp(u, xw_unit, psi, left=0.0, right=0.0)
        rows.append(env * vals)

    return normalize_rows(np.array(rows, dtype=float))


def make_basis(t: np.ndarray, n_modes: int, basis_name: str) -> np.ndarray:
    if basis_name == "fourier":
        return fourier_basis(t, n_modes)

    if basis_name == "bspline_local":
        return bspline_basis(t, n_modes)

    if basis_name == "hybrid_fourier_bspline":
        return hybrid_fourier_bspline_basis(t, n_modes)

    if basis_name == "chebyshev":
        return chebyshev_basis(t, n_modes)

    if basis_name == "wavelet_db4":
        return wavelet_basis(t, n_modes, "db4")

    raise ValueError(f"Unknown basis_name = {basis_name}")


# =============================================================================
# Path reconstruction and action
# =============================================================================

def reconstruct_path(
    false: np.ndarray,
    true: np.ndarray,
    t: np.ndarray,
    coeffs: np.ndarray,
    n_modes: int,
    basis_name: str,
):
    B = make_basis(t, n_modes, basis_name)
    n_basis = B.shape[0]
    Cmat = coeffs.reshape(2, n_basis)
    straight = false[None, :] + t[:, None] * (true - false)[None, :]
    path = straight + B.T @ Cmat.T
    return path, straight


def omega_d(d: float) -> float:
    return 2.0 * np.pi ** (0.5 * d) / gamma(0.5 * d)


def action_prefactor(d: float) -> float:
    return ((d - 1.0) ** (d - 1.0)) * omega_d(d) / d


def path_action(
    coeffs: np.ndarray,
    false: np.ndarray,
    true: np.ndarray,
    Vfalse: float,
    Vtrue: float,
    t: np.ndarray,
    n_modes: int,
    basis_name: str,
) -> float:
    path, _ = reconstruct_path(false, true, t, coeffs, n_modes, basis_name)

    Vpath = V(path) - Vfalse

    # Smooth monotonic tunneling-potential interpolation.
    # Since Vtrue < Vfalse, this decreases from 0 to Vtrue-Vfalse.
    Vt = t * t * (3.0 - 2.0 * t) * (Vtrue - Vfalse)

    dVt = Vt[1:] - Vt[:-1]
    minus_dVt = -dVt

    dphi = path[1:] - path[:-1]
    ds = np.linalg.norm(dphi, axis=1)

    Vdiff = Vpath[:-1] + Vpath[1:] - Vt[:-1] - Vt[1:]

    # Keep the path in a physically reasonable plotting box.
    xmin, xmax = -0.15, 1.70
    ymin, ymax = -0.15, 1.70
    outside = (
        np.maximum(xmin - path[:, 0], 0.0) ** 2
        + np.maximum(path[:, 0] - xmax, 0.0) ** 2
        + np.maximum(ymin - path[:, 1], 0.0) ** 2
        + np.maximum(path[:, 1] - ymax, 0.0) ** 2
    )
    box_penalty = 1.0e8 * np.sum(outside)

    if np.any(Vdiff <= 0.0) or np.any(minus_dVt <= 0.0) or not np.all(np.isfinite(Vdiff)):
        bad1 = np.minimum(Vdiff, 0.0)
        bad2 = np.minimum(minus_dVt, 0.0)
        return float(1e12 + box_penalty + 1e12 * (np.sum(bad1 * bad1) + np.sum(bad2 * bad2)))

    terms = (Vdiff ** (0.5 * ACTION_DIM)) * (ds ** ACTION_DIM) / (minus_dVt ** (ACTION_DIM - 1.0))
    S = action_prefactor(ACTION_DIM) * np.sum(terms)

    if not np.isfinite(S):
        return 1e50

    return float(S + box_penalty)


def optimize_modes(
    false: np.ndarray,
    true: np.ndarray,
    Vfalse: float,
    Vtrue: float,
    basis_name: str,
):
    t = np.linspace(0.0, 1.0, N_GRID)

    results = {}
    previous_coeffs = None
    previous_n = None

    amp = 0.35 * np.linalg.norm(true - false)

    for n_modes in MODES_TO_STORE:
        B = make_basis(t, n_modes, basis_name)
        n_basis = B.shape[0]
        nvar = 2 * n_basis

        starts = []

        # zero start
        starts.append(("zero", np.zeros(nvar)))

        # warm start from previous mode
        if previous_coeffs is not None:
            old = previous_coeffs.reshape(2, previous_n)
            new = np.zeros((2, n_basis))
            m = min(previous_n, n_basis)
            new[:, :m] = old[:, :m]
            starts.append(("warm", new.ravel()))

        # deterministic small random starts
        for j in range(3):
            rng = np.random.default_rng(1000 + 37 * n_modes + j)
            starts.append((f"rand{j + 1}", rng.normal(scale=0.04 * amp, size=nvar)))

        bounds = [(-amp, amp)] * nvar
        best = {"S": np.inf, "coeffs": None, "start": None, "time": None}

        t0 = time.perf_counter()
        for label, x0 in starts:
            # Powell is slower but robust without requiring gradients.
            res = minimize(
                lambda x: path_action(x, false, true, Vfalse, Vtrue, t, n_modes, basis_name),
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": MAXITER, "xtol": 1e-5, "ftol": 1e-8, "disp": False},
            )
            if float(res.fun) < best["S"]:
                best = {
                    "S": float(res.fun),
                    "coeffs": np.asarray(res.x),
                    "start": label,
                    "time": None,
                }

        best["time"] = time.perf_counter() - t0

        path, straight = reconstruct_path(false, true, t, best["coeffs"], n_modes, basis_name)
        best["path"] = path
        best["straight"] = straight
        best["t"] = t
        best["Vpath"] = V(path) - Vfalse
        best["n_modes"] = n_modes
        best["basis_name"] = basis_name

        results[n_modes] = best

        previous_coeffs = best["coeffs"]
        previous_n = n_basis

        print(
            f"{basis_name:24s} n_modes={n_modes:2d}: "
            f"S={best['S']:.8g}, start={best['start']}, time={best['time']:.3f}s"
        )

    return results


# =============================================================================
# Plotting
# =============================================================================

def make_grid(false, true, results):
    paths = [r["path"] for r in results.values()]
    allpts = np.vstack([false[None, :], true[None, :]] + paths)

    xmin, ymin = np.min(allpts, axis=0)
    xmax, ymax = np.max(allpts, axis=0)

    # Add padding and enough room to show the surrounding valley.
    pad_x = max(0.25, 0.10 * (xmax - xmin))
    pad_y = max(0.25, 0.10 * (ymax - ymin))

    xmin = min(xmin - pad_x, -0.15)
    ymin = min(ymin - pad_y, -0.15)
    xmax = max(xmax + pad_x, 1.35)
    ymax = max(ymax + pad_y, 1.35)

    x = np.linspace(xmin, xmax, 320)
    y = np.linspace(ymin, ymax, 320)
    X, Y = np.meshgrid(x, y)
    Z = V(np.stack([X, Y], axis=-1))
    return X, Y, Z


def plot_single_basis_contour(false, true, Vfalse, Vtrue, results, basis_name):
    # Pick the best action among scanned modes.
    best_n = min(results.keys(), key=lambda n: results[n]["S"])
    best = results[best_n]

    # Use this one path to set plot window.
    X, Y, Z = make_grid(false, true, {best_n: best})

    # Safety in case the grid has any bad values
    if np.any(~np.isfinite(Z)):
        print("Warning: non-finite values found in Z. Masking them for plotting.")
        Z = np.where(np.isfinite(Z), Z, np.nan)

    # Slightly wider because legend will go outside
    fig, ax = plt.subplots(figsize=(10.6, 6.4))

    # Use full finite range to avoid white holes/patched corners
    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)

    levels_f = np.linspace(zmin, zmax, 48)   # filled contours
    levels_c = np.linspace(zmin, zmax, 30)   # contour lines

    cf = ax.contourf(
        X, Y, Z,
        levels=levels_f,
        cmap="Spectral_r",
        alpha=0.84,
        extend="both",
    )

    ax.contour(
        X, Y, Z,
        levels=levels_c,
        colors="k",
        linewidths=0.35,
        alpha=0.18,
    )

    cbar = fig.colorbar(
        cf,
        ax=ax,
        pad=0.02,
        fraction=0.048,
    )
    cbar.set_label(r"$V(\phi_1,\phi_2)$")
    cbar.ax.tick_params(labelsize=10)

    # Straight path
    t = best["t"]
    straight = false[None, :] + t[:, None] * (true - false)[None, :]
    ax.plot(
        straight[:, 0],
        straight[:, 1],
        "k--",
        linewidth=2.5,
        label="straight path",
        zorder=6,
    )

    # Best optimized path for this basis
    ax.plot(
        best["path"][:, 0],
        best["path"][:, 1],
        color=basis_color(basis_name),
        linestyle=basis_linestyle(basis_name),
        linewidth=3.0,
        label=fr"{nice_basis_label(basis_name)}, $N={best_n}$, $S={best['S']:.3g}$",
        zorder=7,
    )

    ax.scatter(
        [false[0]], [false[1]],
        s=150,
        color="red",
        edgecolor="black",
        linewidth=0.9,
        zorder=10,
        label="false",
    )

    ax.scatter(
        [true[0]], [true[1]],
        s=150,
        color="green",
        edgecolor="black",
        linewidth=0.9,
        zorder=10,
        label="true",
    )

    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_title(fr"{nice_basis_label(basis_name)} path in a two-field OptiBounce potential")

    style_axes(ax)

    # Put legend outside so it does not clash with curves/contours
    ax.legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        frameon=True,
    )

    # Leave space on the right for colorbar + legend
    fig.subplots_adjust(right=0.76)

    safe_name = basis_name.replace("/", "_")
    out = OUTDIR / f"optibounce2d_{safe_name}_path_contour.png"
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


def plot_convergence(results, basis_name: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    n = np.array(sorted(results.keys()))
    S = np.array([results[k]["S"] for k in n])

    ax.plot(
        n,
        S,
        marker="o",
        color=basis_color(basis_name),
        linewidth=2.4,
        label=nice_basis_label(basis_name),
    )
    ax.axhline(
        np.min(S),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=fr"best $S={np.min(S):.3g}$",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"number of basis functions $N$")
    ax.set_ylabel(r"action $S_4$")
    ax.set_title(fr"Convergence of {nice_basis_label(basis_name)} deformation")
    style_axes(ax)
    ax.legend()

    safe_name = basis_name.replace("/", "_")
    out = OUTDIR / f"optibounce2d_{safe_name}_convergence.png"
    fig.tight_layout()
    fig.savefig(out)
    print("Saved", out)


def plot_profiles(false, true, results, basis_name: str):
    # Use best mode, not necessarily highest mode.
    nbest = min(results.keys(), key=lambda n: results[n]["S"])
    r = results[nbest]
    t = r["t"]
    path = r["path"]
    straight = r["straight"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.plot(t, path[:, 0], color="#1f4aff", linewidth=2.5, label=fr"$\phi_1(t)$ optimized")
    ax.plot(t, path[:, 1], color="#c7361d", linewidth=2.5, label=fr"$\phi_2(t)$ optimized")

    ax.plot(t, straight[:, 0], color="#1f4aff", linestyle="--", alpha=0.6, label=fr"$\phi_1(t)$ straight")
    ax.plot(t, straight[:, 1], color="#c7361d", linestyle="--", alpha=0.6, label=fr"$\phi_2(t)$ straight")

    ax.set_xlabel(r"path parameter $t$")
    ax.set_ylabel(r"field value")
    ax.set_title(fr"{nice_basis_label(basis_name)} optimized profiles, $N={nbest}$")
    style_axes(ax)
    ax.legend(fontsize=8, ncol=2)

    safe_name = basis_name.replace("/", "_")
    out = OUTDIR / f"optibounce2d_{safe_name}_profiles.png"
    fig.tight_layout()
    fig.savefig(out)
    print("Saved", out)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="2D OptiBounce basis-path visualization.")
    parser.add_argument(
        "--basis",
        default="all",
        choices=["all"] + BASIS_LIST,
        help="Basis to plot. Use 'all' to make one separate plot per basis.",
    )
    parser.add_argument(
        "--extra-plots",
        action="store_true",
        help="Also save convergence/profile plots for every basis. By default, these are saved only for Fourier.",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    false = np.array([0.0, 0.0], dtype=float)
    Vfalse = float(V(false))
    true, Vtrue = find_true_vacuum()

    H0 = hessian_at_origin()
    print("2D OptiBounce-like potential")
    print("C =", C, "DELTA =", DELTA)
    print("false =", false, "Vfalse =", Vfalse)
    print("true  =", true, "Vtrue  =", Vtrue)
    print("DeltaV =", Vfalse - Vtrue)
    print("Hessian at origin =", H0)
    print("origin is local minimum:", np.all(np.linalg.eigvalsh(H0) > 0))

    if Vfalse <= Vtrue:
        raise RuntimeError("False vacuum is not above true vacuum.")

    if args.basis == "all":
        bases_to_run = BASIS_LIST
    else:
        bases_to_run = [args.basis]

    for basis_name in bases_to_run:
        print("\n" + "=" * 100)
        print("Running basis:", basis_name)
        print("=" * 100)

        results = optimize_modes(false, true, Vfalse, Vtrue, basis_name)

        plot_single_basis_contour(false, true, Vfalse, Vtrue, results, basis_name)

        # Keep default output light: only Fourier gets convergence/profile plots.
        if basis_name == "fourier" or args.extra_plots:
            plot_convergence(results, basis_name)
            plot_profiles(false, true, results, basis_name)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
