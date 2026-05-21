#!/usr/bin/env python3
"""
plot_ct_preconditioning_summary.py

Create clean paper-style plots for the CosmoTransitions preconditioning test.

This script reads the median summary CSV produced by fourier_ct_scan_repeats.py
and makes TWO SEPARATE figures:

    1. ct_preconditioning_steps.png
       Median CosmoTransitions deformation steps vs number of fields.

    2. ct_preconditioning_time.png
       Median total wall time vs number of fields.

The comparison shown is:

    - straight-line initialization
    - Fourier-preconditioned initialization with m = 1

The script does not rerun CosmoTransitions. It only reads the saved summary CSV.

Expected input columns in scan_summary.csv
------------------------------------------
nfields
nmodes
method
steps_median
total_median

Optional columns used if present:
fRatio_median
action_median

Example
-------
python3 plot_ct_preconditioning_summary.py \
    --summary scan_N2_to_N10_summary.csv \
    --output-prefix ct_preconditioning \
    --min-nfields 3

This produces:
    ct_preconditioning_steps.png
    ct_preconditioning_steps.pdf
    ct_preconditioning_time.png
    ct_preconditioning_time.pdf
    ct_preconditioning_reductions.csv

Author: Aadarsh Singh et al.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Plot style
# ============================================================

def apply_paper_style() -> None:
    """Apply a clean matplotlib style suitable for paper figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.facecolor": "#fbfbfb",
        "figure.facecolor": "white",
        "axes.edgecolor": "#2b2b2b",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.facecolor": "white",
        "legend.edgecolor": "#d0d0d0",
        "grid.color": "#b0b0b0",
        "grid.alpha": 0.28,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.6,
        "lines.markersize": 8.0,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def style_axes(ax: plt.Axes) -> None:
    """Consistent axis styling."""
    ax.grid(True, which="both")
    ax.tick_params(direction="in", top=True, right=True, length=4)
    for spine in ax.spines.values():
        spine.set_color("#2b2b2b")


# ============================================================
# Data handling
# ============================================================

REQUIRED_COLUMNS = {
    "nfields",
    "nmodes",
    "method",
    "steps_median",
    "total_median",
}


def load_summary(path: str | Path) -> pd.DataFrame:
    """Load and validate scan_summary.csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input summary file not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["nfields"] = pd.to_numeric(df["nfields"], errors="raise")
    df["steps_median"] = pd.to_numeric(df["steps_median"], errors="coerce")
    df["total_median"] = pd.to_numeric(df["total_median"], errors="coerce")

    return df


def normalize_nmodes(series: pd.Series) -> pd.Series:
    """
    Convert nmodes column to a string form useful for matching.

    In the scan summary, straight paths use nmodes='baseline', while Fourier
    paths use integer mode numbers such as 1, 2, 3. Depending on CSV parsing,
    these can appear as strings or numbers.
    """
    return series.astype(str).str.strip().str.lower()


def select_straight_and_fourier_m1(
    df: pd.DataFrame,
    min_nfields: int | None = None,
    max_nfields: int | None = None,
) -> pd.DataFrame:
    """
    Select straight-line baseline and Fourier m=1 rows.

    Returns a tidy dataframe with columns:
        nfields, init, steps, total_time
    """
    tmp = df.copy()
    tmp["method_norm"] = tmp["method"].astype(str).str.strip().str.lower()
    tmp["nmodes_norm"] = normalize_nmodes(tmp["nmodes"])

    if min_nfields is not None:
        tmp = tmp[tmp["nfields"] >= min_nfields]
    if max_nfields is not None:
        tmp = tmp[tmp["nfields"] <= max_nfields]

    straight = tmp[tmp["method_norm"].eq("straight")].copy()
    fourier_m1 = tmp[
        tmp["method_norm"].eq("fourier_jax")
        & tmp["nmodes_norm"].isin(["1", "1.0"])
    ].copy()

    if straight.empty:
        raise ValueError("No straight baseline rows found: method == 'straight'.")

    if fourier_m1.empty:
        raise ValueError(
            "No Fourier m=1 rows found: method == 'fourier_jax' and nmodes == 1."
        )

    straight_tidy = pd.DataFrame({
        "nfields": straight["nfields"].astype(int),
        "init": "Straight",
        "steps": straight["steps_median"],
        "total_time": straight["total_median"],
    })

    fourier_tidy = pd.DataFrame({
        "nfields": fourier_m1["nfields"].astype(int),
        "init": r"Fourier $m=1$",
        "steps": fourier_m1["steps_median"],
        "total_time": fourier_m1["total_median"],
    })

    tidy = pd.concat([straight_tidy, fourier_tidy], ignore_index=True)
    tidy = tidy.sort_values(["nfields", "init"]).reset_index(drop=True)

    # Check that both methods exist for each displayed N.
    counts = tidy.groupby("nfields")["init"].nunique()
    bad = counts[counts < 2]
    if not bad.empty:
        raise ValueError(
            "Some nfields values do not have both straight and Fourier m=1 rows: "
            + ", ".join(map(str, bad.index.tolist()))
        )

    return tidy


def compute_reductions(tidy: pd.DataFrame) -> pd.DataFrame:
    """Compute step and total-time reductions for Fourier m=1 vs straight."""
    rows = []
    for nfields, group in tidy.groupby("nfields"):
        straight = group[group["init"].eq("Straight")].iloc[0]
        fourier = group[group["init"].str.contains("Fourier", regex=False)].iloc[0]

        step_red = 100.0 * (straight["steps"] - fourier["steps"]) / straight["steps"]
        time_red = 100.0 * (
            straight["total_time"] - fourier["total_time"]
        ) / straight["total_time"]

        rows.append({
            "nfields": int(nfields),
            "straight_steps": float(straight["steps"]),
            "fourier_steps": float(fourier["steps"]),
            "step_reduction_percent": float(step_red),
            "straight_total_time": float(straight["total_time"]),
            "fourier_total_time": float(fourier["total_time"]),
            "time_reduction_percent": float(time_red),
        })

    return pd.DataFrame(rows).sort_values("nfields").reset_index(drop=True)


# ============================================================
# Plotting helpers
# ============================================================

METHODS = ["Straight", r"Fourier $m=1$"]
COLORS = {
    "Straight": "#222222",
    r"Fourier $m=1$": "#1f4aff",
}
MARKERS = {
    "Straight": "s",
    r"Fourier $m=1$": "o",
}
LINESTYLES = {
    "Straight": "--",
    r"Fourier $m=1$": "-",
}


def plot_quantity(
    tidy: pd.DataFrame,
    reductions: pd.DataFrame,
    quantity: str,
    ylabel: str,
    title: str,
    output: str | Path,
    save_pdf: bool = True,
    annotate: bool = True,
) -> None:
    """Plot one quantity, either 'steps' or 'total_time'."""
    if quantity not in {"steps", "total_time"}:
        raise ValueError("quantity must be either 'steps' or 'total_time'.")

    output = Path(output)
    fig, ax = plt.subplots(figsize=(6.7, 4.8))

    for method in METHODS:
        sub = tidy[tidy["init"].eq(method)].sort_values("nfields")
        ax.plot(
            sub["nfields"],
            sub[quantity],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            color=COLORS[method],
            label=method,
        )

    ax.set_xlabel(r"number of fields $N_\phi$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(tidy["nfields"].unique()))
    style_axes(ax)
    ax.legend(loc="best")

    if annotate and not reductions.empty:
        if quantity == "steps":
            # Put only one clean annotation, not one label per point.
            nmin = reductions["nfields"].min()
            nmax = reductions["nfields"].max()
            mean_red = reductions["step_reduction_percent"].mean()
            text = rf"average reduction: {mean_red:.0f}%"
            ax.text(
                0.63,
                0.08,
                text,
                transform=ax.transAxes,
                color=COLORS[r"Fourier $m=1$"],
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d0d0d0", alpha=0.92),
            )
        else:
            mean_red = reductions["time_reduction_percent"].mean()
            min_red = reductions["time_reduction_percent"].min()
            max_red = reductions["time_reduction_percent"].max()
            text = rf"runtime gain: {min_red:.0f}-{max_red:.0f}%"
            ax.text(
                0.63,
                0.08,
                text,
                transform=ax.transAxes,
                color=COLORS[r"Fourier $m=1$"],
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d0d0d0", alpha=0.92),
            )

    fig.tight_layout()
    fig.savefig(output)
    print(f"Saved {output}")

    if save_pdf:
        pdf_path = output.with_suffix(".pdf")
        fig.savefig(pdf_path)
        print(f"Saved {pdf_path}")

    plt.close(fig)


def make_plots(
    tidy: pd.DataFrame,
    reductions: pd.DataFrame,
    output_prefix: str | Path,
    save_pdf: bool = True,
    annotate: bool = True,
) -> None:
    """Create two separate paper figures."""
    apply_paper_style()
    output_prefix = Path(output_prefix)

    steps_output = output_prefix.with_name(output_prefix.name + "_steps.png")
    time_output = output_prefix.with_name(output_prefix.name + "_time.png")

    plot_quantity(
        tidy=tidy,
        reductions=reductions,
        quantity="steps",
        ylabel="median deformation steps",
        title="CosmoTransitions deformation work",
        output=steps_output,
        save_pdf=save_pdf,
        annotate=annotate,
    )

    plot_quantity(
        tidy=tidy,
        reductions=reductions,
        quantity="total_time",
        ylabel="median total time [s]",
        title="Total wall time including Fourier preprocessing",
        output=time_output,
        save_pdf=save_pdf,
        annotate=annotate,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Make separate paper plots comparing straight-line and Fourier m=1 "
            "CosmoTransitions initializations."
        )
    )
    parser.add_argument(
        "--summary",
        default="scan_summary.csv",
        help="Input CSV produced by fourier_ct_scan_repeats.py.",
    )
    parser.add_argument(
        "--output-prefix",
        default="ct_preconditioning",
        help=(
            "Output prefix. The script writes '<prefix>_steps.png' and "
            "'<prefix>_time.png'."
        ),
    )
    parser.add_argument(
        "--reductions-csv",
        default="ct_preconditioning_reductions.csv",
        help="Output CSV containing step/time reductions.",
    )
    parser.add_argument(
        "--min-nfields",
        type=int,
        default=None,
        help="Optional minimum N_phi to include, e.g. 3 to omit the special N=2 base case.",
    )
    parser.add_argument(
        "--max-nfields",
        type=int,
        default=None,
        help="Optional maximum N_phi to include.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not also save PDF versions.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Do not add the small summary annotation boxes to the plots.",
    )

    args = parser.parse_args()

    df = load_summary(args.summary)
    tidy = select_straight_and_fourier_m1(
        df,
        min_nfields=args.min_nfields,
        max_nfields=args.max_nfields,
    )
    reductions = compute_reductions(tidy)

    reductions.to_csv(args.reductions_csv, index=False)
    print(f"Saved {args.reductions_csv}")
    print()
    print("Reduction summary:")
    print(reductions.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print()

    make_plots(
        tidy=tidy,
        reductions=reductions,
        output_prefix=args.output_prefix,
        save_pdf=not args.no_pdf,
        annotate=not args.no_annotate,
    )


if __name__ == "__main__":
    main()
