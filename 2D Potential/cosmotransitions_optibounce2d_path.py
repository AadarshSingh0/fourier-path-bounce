#!/usr/bin/env python3
"""
cosmotransitions_optibounce2d_path_check.py

Standalone CosmoTransitions cross-check for the same 2D OptiBounce-like potential
used in the paper-style basis-path visualization script.

Goal
----
Run CosmoTransitions on

    V(phi) = [ c1 (phi1-1)^2 + c2 (phi2-1)^2 - delta ] (phi1^2 + phi2^2)

with

    c = (0.684373, 0.181928), delta = 0.065

and save a contour plot showing the CosmoTransitions deformed path.

This script is independent of the Fourier/B-spline basis script. It does not need
your basis output files.

Important convention
--------------------
CosmoTransitions expects the initial path array ordered as

    true vacuum -> false vacuum

whereas our basis scripts often plot

    false vacuum -> true vacuum.

Here:
    false = (0, 0)
    true  = found numerically by minimizing V

Outputs
-------
plots/optibounce2d/cosmotransitions/
    ct_optibounce2d_path_contour.png
    ct_optibounce2d_path_V.png
    ct_optibounce2d_path.csv

Run
---
    python3 cosmotransitions_optibounce2d_path_check.py

No pop-up:
    python3 cosmotransitions_optibounce2d_path_check.py --no-show

Try more/fewer path points:
    python3 cosmotransitions_optibounce2d_path_check.py --npts 120 --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =============================================================================
# Potential settings
# =============================================================================

C = np.array([0.684373, 0.181928], dtype=float)
DELTA = 0.065

OUTDIR = Path("plots") / "optibounce2d" / "cosmotransitions"


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
# Potential and derivatives
# =============================================================================

def V(phi: np.ndarray) -> np.ndarray:
    """
    Potential accepts either shape (2,) or (..., 2).
    """
    phi = np.asarray(phi, dtype=float)
    A = np.sum(C * (phi - 1.0) ** 2, axis=-1) - DELTA
    R2 = np.sum(phi ** 2, axis=-1)
    return A * R2


def gradV(phi: np.ndarray) -> np.ndarray:
    """
    Gradient accepts shape (2,) or (N, 2).
    CosmoTransitions may call it on arrays of path points.
    """
    phi = np.asarray(phi, dtype=float)

    if phi.ndim == 1:
        A = np.sum(C * (phi - 1.0) ** 2) - DELTA
        R2 = np.sum(phi ** 2)
        return 2.0 * C * (phi - 1.0) * R2 + 2.0 * phi * A

    A = np.sum(C[None, :] * (phi - 1.0) ** 2, axis=1) - DELTA
    R2 = np.sum(phi ** 2, axis=1)
    return 2.0 * C[None, :] * (phi - 1.0) * R2[:, None] + 2.0 * phi * A[:, None]


def hessian_at_origin() -> np.ndarray:
    A0 = np.sum(C) - DELTA
    return 2.0 * A0 * np.eye(2)


def find_true_vacuum() -> tuple[np.ndarray, float]:
    """
    False vacuum is fixed at origin. True vacuum is found numerically.
    """
    guesses = []
    for x in np.linspace(-0.4, 1.8, 9):
        for y in np.linspace(-0.4, 1.8, 9):
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


# =============================================================================
# CosmoTransitions run
# =============================================================================

def run_cosmotransitions(false: np.ndarray, true: np.ndarray, npts: int):
    try:
        from cosmoTransitions import pathDeformation as pd
    except ImportError as exc:
        raise ImportError(
            "Could not import CosmoTransitions. Activate your cosmo_env or install it."
        ) from exc

    # CosmoTransitions convention: true/lower minimum first, false/metastable last.
    path_pts = np.linspace(true, false, npts)

    print("\nRunning CosmoTransitions fullTunneling")
    print("Initial path order: true -> false")
    print("path_pts shape =", path_pts.shape)

    Y = pd.fullTunneling(
        path_pts,
        V,
        gradV,
        maxiter=30,
        verbose=True,
        V_spline_samples=200,

        # O(4) zero-temperature bounce.
        # This is passed to SingleFieldInstanton initialization.
        tunneling_init_params={
            "alpha": 3,
        },

        deformation_deform_params={
            "fRatioConv": 0.02,
            "maxiter": 500,
            "verbose": 1,
        },
    )

    print("\nCosmoTransitions result")
    print("  action =", getattr(Y, "action", None))
    print("  fRatio =", getattr(Y, "fRatio", None))
    print("  Phi shape =", getattr(Y, "Phi", np.array([])).shape)

    return Y


# =============================================================================
# Plotting
# =============================================================================

def make_grid(false: np.ndarray, true: np.ndarray, ct_path: np.ndarray):
    allpts = np.vstack([false[None, :], true[None, :], ct_path])

    xmin, ymin = np.min(allpts, axis=0)
    xmax, ymax = np.max(allpts, axis=0)

    pad_x = max(0.25, 0.10 * (xmax - xmin))
    pad_y = max(0.25, 0.10 * (ymax - ymin))

    xmin = min(xmin - pad_x, -0.15)
    ymin = min(ymin - pad_y, -0.15)
    xmax = max(xmax + pad_x, 1.35)
    ymax = max(ymax + pad_y, 1.35)

    x = np.linspace(xmin, xmax, 340)
    y = np.linspace(ymin, ymax, 340)
    X, Y = np.meshgrid(x, y)
    Z = V(np.stack([X, Y], axis=-1))
    return X, Y, Z


def plot_ct_path(false: np.ndarray, true: np.ndarray, Yct):
    """
    CosmoTransitions returns Yct.Phi ordered true -> false.
    For visual comparison, we plot it as returned; endpoints are labelled.

    Improvements:
      1. Legend is moved outside the plotting area.
      2. contourf uses the full finite Z range, so no white holes appear near
         the true vacuum or in the plot corners.
      3. extend='both' is used to color any values outside the contour levels.
    """
    ct_path = np.asarray(Yct.Phi, dtype=float)

    X, Y, Z = make_grid(false, true, ct_path)

    # Safety: replace non-finite values if any appear.
    if np.any(~np.isfinite(Z)):
        print("Warning: non-finite values found in Z. Masking them for plotting.")
        Z = np.where(np.isfinite(Z), Z, np.nan)

    # Wider figure because legend is outside.
    fig, ax = plt.subplots(figsize=(10.6, 6.4))

    # Use full finite range to avoid white patches.
    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)

    # Filled contours and line contours can use different densities.
    levels_f = np.linspace(zmin, zmax, 48)
    levels_c = np.linspace(zmin, zmax, 30)

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

    # Straight path false -> true for visual reference.
    straight = np.linspace(false, true, 300)
    ax.plot(
        straight[:, 0],
        straight[:, 1],
        "k--",
        linewidth=2.5,
        label="straight path",
        zorder=6,
    )

    action = getattr(Yct, "action", np.nan)
    fRatio = getattr(Yct, "fRatio", np.nan)

    # CosmoTransitions path. Direction is true -> false, but visually this is fine.
    ax.plot(
        ct_path[:, 0],
        ct_path[:, 1],
        color="#111111",
        linestyle="-",
        linewidth=3.1,
        alpha=0.92,
        label=fr"CosmoTransitions, $S={action:.3g}$, $f_R={fRatio:.2g}$",
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
    ax.set_title("CosmoTransitions path in a two-field OptiBounce potential")

    style_axes(ax)

    # Legend outside, to the right of the colorbar.
    ax.legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        frameon=True,
    )

    # Leave room on the right for colorbar + legend.
    fig.subplots_adjust(right=0.76)

    out = OUTDIR / "ct_optibounce2d_path_contour.png"
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


def plot_V_along_path(false: np.ndarray, true: np.ndarray, Yct):
    ct_path = np.asarray(Yct.Phi, dtype=float)
    s = np.linspace(0.0, 1.0, len(ct_path))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.plot(
        s,
        V(ct_path),
        color="#111111",
        linewidth=2.5,
        label="CosmoTransitions path",
    )

    straight = np.linspace(true, false, len(ct_path))
    ax.plot(
        s,
        V(straight),
        color="black",
        linestyle="--",
        alpha=0.65,
        linewidth=2.0,
        label="straight path",
    )

    ax.axhline(float(V(false)), color="red", linestyle=":", linewidth=1.5, label=r"$V_{\rm false}$")
    ax.axhline(float(V(true)), color="green", linestyle=":", linewidth=1.5, label=r"$V_{\rm true}$")

    ax.set_xlabel("path parameter")
    ax.set_ylabel(r"$V(\phi)$")
    ax.set_title("Potential along CosmoTransitions path")
    style_axes(ax)
    ax.legend(fontsize=8)

    out = OUTDIR / "ct_optibounce2d_path_V.png"
    fig.tight_layout()
    fig.savefig(out)
    print("Saved", out)


def save_path_csv(Yct):
    path = np.asarray(Yct.Phi, dtype=float)
    out = OUTDIR / "ct_optibounce2d_path.csv"

    data = np.column_stack([
        np.arange(len(path)),
        path[:, 0],
        path[:, 1],
        V(path),
    ])

    header = "index,phi1,phi2,V"
    np.savetxt(out, data, delimiter=",", header=header, comments="")
    print("Saved", out)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone CosmoTransitions check for 2D OptiBounce-like potential.")
    parser.add_argument("--npts", type=int, default=80, help="Number of points in the initial true->false path.")
    parser.add_argument("--no-show", action="store_true", help="Save plots but do not open interactive windows.")
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

    Yct = run_cosmotransitions(false, true, args.npts)

    plot_ct_path(false, true, Yct)
    plot_V_along_path(false, true, Yct)
    save_path_csv(Yct)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
