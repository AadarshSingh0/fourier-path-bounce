#!/usr/bin/env python3
"""
plot_basis_comparison_from_csv_cli.py

Make basis-comparison plots from:

    results/<RUN_LABEL>/basis_comparison_summary.csv

Examples
--------
python3 plot_basis_comparison_from_csv_cli.py --run-label optibounce_D3
python3 plot_basis_comparison_from_csv_cli.py --run-label optibounce_D4
python3 plot_basis_comparison_from_csv_cli.py --run-label mega_random_D4

Optional:
python3 plot_basis_comparison_from_csv_cli.py --run-label optibounce_D3 --nphi-min 5 --nphi-max 20
python3 plot_basis_comparison_from_csv_cli.py --csv results/optibounce_D3/basis_comparison_summary.csv --outdir plots/test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_RUN_LABEL = "optibounce_D3"

LABEL_MAP = {
    "fourier": "Fourier",
    "chebyshev": "Chebyshev",
    "legendre": "Legendre",
    "bspline_local": "B-spline local",
    "gaussian_local": "Gaussian local",
    "hybrid_fourier_gaussian": "Fourier+Gaussian",
    "hybrid_fourier_bspline": "Fourier+B-spline",
    "wavelet_db2": "Wavelet db2",
    "wavelet_db4": "Wavelet db4",
    "wavelet_db6": "Wavelet db6",
    "wavelet_sym4": "Wavelet sym4",
    "wavelet_coif1": "Wavelet coif1",
}

GOOD_BASES = [
    "Fourier",
    "B-spline local",
    "Fourier+Gaussian",
    "Fourier+B-spline",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make basis-comparison plots from saved CSV summary."
    )
    parser.add_argument(
        "--run-label",
        default=DEFAULT_RUN_LABEL,
        help="Folder name inside results/ and plots/, e.g. optibounce_D3, optibounce_D4, mega_random_D4.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional direct path to basis_comparison_summary.csv. Overrides --run-label for input.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output plot folder. Default: plots/<RUN_LABEL>/.",
    )
    parser.add_argument(
        "--nphi-min",
        type=int,
        default=None,
        help="Minimum nphi to plot. Default: 5 for optibounce_D3/D4, otherwise no lower cut.",
    )
    parser.add_argument(
        "--nphi-max",
        type=int,
        default=None,
        help="Maximum nphi to plot. Default: 20 for optibounce_D3/D4, otherwise no upper cut.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open/show plots interactively; only save files.",
    )
    return parser.parse_args()


def has_useful_numeric(df, col):
    if col not in df.columns:
        return False
    vals = pd.to_numeric(df[col], errors="coerce")
    return vals.notna().any()


def make_series_with_gaps(sub, xcol, ycol):
    """
    Insert NaN for missing integer nphi values, so lines break across gaps.
    """
    sub = sub[[xcol, ycol]].copy()
    sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
    sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
    sub = sub.dropna(subset=[xcol])
    if sub.empty:
        return sub

    sub[xcol] = sub[xcol].astype(int)
    sub = sub.sort_values(xcol).groupby(xcol, as_index=False)[ycol].first()

    full = pd.DataFrame({xcol: np.arange(int(sub[xcol].min()), int(sub[xcol].max()) + 1)})
    return full.merge(sub, on=xcol, how="left")


def plot_metric(df, plot_dir, metric, ylabel, title, filename, ylog=False, bases=None, show=True):
    if not has_useful_numeric(df, metric):
        print(f"Skipping {filename}: column {metric} has no usable numeric data.")
        return

    plt.figure(figsize=(7.4, 5.0))

    if bases is None:
        bases = list(df["basis_label"].dropna().unique())

    for i, basis in enumerate(bases):
        sub = df[df["basis_label"] == basis]
        if sub.empty:
            continue

        sub_gap = make_series_with_gaps(sub, "nphi", metric)
        if sub_gap.empty or sub_gap[metric].isna().all():
            continue

        plt.plot(
            sub_gap["nphi"],
            sub_gap[metric],
            marker=MARKERS[i % len(MARKERS)],
            linewidth=1.8,
            markersize=5,
            label=basis,
        )

    if ylog:
        plt.yscale("log")

    plt.xlabel(r"$N_\phi$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_dir / filename, dpi=300)
    plt.savefig(plot_dir / filename.replace(".png", ".pdf"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_action(df, plot_dir, show=True):
    if not has_useful_numeric(df, "best_S"):
        print("Skipping action plot: best_S has no usable numeric data.")
        return

    plt.figure(figsize=(7.4, 5.0))

    if has_useful_numeric(df, "reference_FB"):
        ref = df[["nphi", "reference_FB"]].drop_duplicates().sort_values("nphi")
        ref_gap = make_series_with_gaps(ref, "nphi", "reference_FB")
        plt.plot(
            ref_gap["nphi"],
            ref_gap["reference_FB"],
            "k--",
            marker="x",
            linewidth=2.0,
            markersize=6,
            label="FindBounce reference",
        )

    bases = list(df["basis_label"].dropna().unique())
    for i, basis in enumerate(bases):
        sub = df[df["basis_label"] == basis]
        sub_gap = make_series_with_gaps(sub, "nphi", "best_S")
        if sub_gap.empty or sub_gap["best_S"].isna().all():
            continue

        plt.plot(
            sub_gap["nphi"],
            sub_gap["best_S"],
            marker=MARKERS[i % len(MARKERS)],
            linewidth=1.5,
            markersize=4.5,
            alpha=0.85,
            label=basis,
        )

    plt.xlabel(r"$N_\phi$")
    plt.ylabel(r"$S$")
    plt.title("Best action reached by different bases")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_dir / "basis_action_vs_nphi.png", dpi=300)
    plt.savefig(plot_dir / "basis_action_vs_nphi.pdf")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    args = parse_args()

    run_label = args.run_label
    csv_path = Path(args.csv) if args.csv is not None else Path("results") / run_label / "basis_comparison_summary.csv"
    plot_dir = Path(args.outdir) if args.outdir is not None else Path("plots") / run_label
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run label: {run_label}")
    print(f"Input CSV: {csv_path}")
    print(f"Plot dir : {plot_dir}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find input CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["nphi", "basis", "best_S", "cumulative_total_time_s", "best_n_basis"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Default cuts: skip the known N_phi=4 OptiBounce caveat for clean basis plots.
    nphi_min = args.nphi_min
    nphi_max = args.nphi_max
    if nphi_min is None and run_label.startswith("optibounce"):
        nphi_min = 5
    if nphi_max is None and run_label.startswith("optibounce"):
        nphi_max = 20

    df["nphi"] = pd.to_numeric(df["nphi"], errors="coerce")
    df = df.dropna(subset=["nphi"]).copy()
    df["nphi"] = df["nphi"].astype(int)

    if nphi_min is not None:
        df = df[df["nphi"] >= nphi_min].copy()
    if nphi_max is not None:
        df = df[df["nphi"] <= nphi_max].copy()

    df["basis_label"] = df["basis"].map(LABEL_MAP).fillna(df["basis"])

    for col in ["best_S", "reference_FB", "rel_to_FB_percent", "cumulative_total_time_s", "best_n_basis"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["basis_label", "nphi"])

    print(f"Rows after cuts: {len(df)}")
    print(f"Nphi range: {df['nphi'].min()} to {df['nphi'].max()}")
    print("Bases:", ", ".join(sorted(df["basis_label"].unique())))

    show = not args.no_show

    plot_metric(
        df, plot_dir,
        metric="cumulative_total_time_s",
        ylabel="cumulative total time [s]",
        title="Runtime comparison for different path bases",
        filename="basis_time_vs_nphi.png",
        ylog=True,
        show=show,
    )

    plot_metric(
        df, plot_dir,
        metric="rel_to_FB_percent",
        ylabel=r"$|S-S_{\rm FB}|/S_{\rm FB}$ [%]",
        title="Relative action error for different path bases",
        filename="basis_relative_error_vs_nphi.png",
        ylog=True,
        show=show,
    )

    plot_metric(
        df, plot_dir,
        metric="rel_to_FB_percent",
        ylabel=r"$|S-S_{\rm FB}|/S_{\rm FB}$ [%]",
        title="Relative action error: competitive bases only",
        filename="basis_relative_error_good_bases.png",
        ylog=True,
        bases=GOOD_BASES,
        show=show,
    )

    plot_action(df, plot_dir, show=show)

    plot_metric(
        df, plot_dir,
        metric="best_n_basis",
        ylabel=r"best $N_{\rm basis}$",
        title="Adaptive basis size selected by no-improvement criterion",
        filename="basis_modes_vs_nphi.png",
        ylog=False,
        show=show,
    )

    print("\nSaved:")
    for name in [
        "basis_time_vs_nphi.png",
        "basis_time_vs_nphi.pdf",
        "basis_relative_error_vs_nphi.png",
        "basis_relative_error_vs_nphi.pdf",
        "basis_relative_error_good_bases.png",
        "basis_relative_error_good_bases.pdf",
        "basis_action_vs_nphi.png",
        "basis_action_vs_nphi.pdf",
        "basis_modes_vs_nphi.png",
        "basis_modes_vs_nphi.pdf",
    ]:
        f = plot_dir / name
        if f.exists():
            print(f"  {f}")


if __name__ == "__main__":
    main()
