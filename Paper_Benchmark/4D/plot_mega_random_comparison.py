#!/usr/bin/env python3
"""
plot_mega_random_comparison.py

Plot action and runtime comparison for the mega-random benchmark using:

    mega_random_jax_summary.csv
    mega_random_cosmotransitions_results.csv
    mega_random_findbounce_results.csv

The script is robust to two FindBounce CSV formats:

1. Normal CSV with columns:
       nphi, status, ActionFindBounceD4, TimeSeconds, ...

2. Mathematica rule-style CSV where each cell looks like:
       "nphi" -> 2
       "ActionFindBounceD4" -> 18.04
       "TimeSeconds" -> 0.168
   This can happen when Mathematica exports a list of Associations/Rules in a
   non-standard way.

Missing intermediate N_phi values are kept as gaps in the plotted lines.
For example, if N_phi=40 and N_phi=42 exist but N_phi=41 is missing,
the curve is not connected across the missing point.

Outputs are saved in:

    plots/mega_random_action_vs_nphi.png/pdf
    plots/mega_random_time_vs_nphi.png/pdf
    plots/mega_random_merged_summary.csv

Run:

    python3 plot_mega_random_comparison.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


JAX_CSV = "mega_random_jax_summary.csv"
CT_CSV = "mega_random_cosmotransitions_results.csv"
FB_CSV = "mega_random_findbounce_results.csv"

OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)


def pick_column(df: pd.DataFrame, candidates, required: bool = True):
    """Return first matching column name from a list of candidate names."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns are: {list(df.columns)}"
        )
    return None


def to_number(x):
    """Convert strings/numbers/Missing[...] safely to float, else NaN."""
    if x is None:
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip().strip('"').strip("'")

    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    if "Missing" in s or "$Failed" in s:
        return np.nan

    # Mathematica sometimes writes *^ for powers.
    s = s.replace("*^", "e")

    try:
        return float(s)
    except Exception:
        return np.nan


def parse_rule_cell(cell: str):
    """
    Parse one Mathematica rule-style CSV cell like:
        '"nphi" -> 2'
        '"status" -> "ok"'
        '"ActionFindBounceD4" -> Missing["Failed"]'
    """
    s = str(cell).strip()

    if "->" not in s:
        return None, None

    key, val = s.split("->", 1)
    key = key.strip().strip('"').strip("'")
    val = val.strip()

    # Remove outer quotes for simple strings.
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        val = val[1:-1]

    return key, val


def read_mathematica_rule_csv(csvfile: str | Path) -> pd.DataFrame:
    """
    Read a Mathematica-exported rule/association CSV.

    This handles files where each row is a list of cells:
        "nphi" -> 2, "status" -> "ok", ...
    with no normal CSV header.
    """
    rows = []

    with open(csvfile, newline="") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            parsed = {}
            for cell in raw_row:
                key, val = parse_rule_cell(cell)
                if key is not None:
                    parsed[key] = val

            if parsed:
                rows.append(parsed)

    if not rows:
        raise ValueError(f"Could not parse Mathematica rule-style CSV: {csvfile}")

    return pd.DataFrame(rows)


def read_normal_or_rule_csv(csvfile: str | Path) -> pd.DataFrame:
    """
    Try normal pandas read first. If the result looks like a Mathematica
    rule-style export, re-read using the rule parser.
    """
    csvfile = Path(csvfile)

    # Peek first nonempty line.
    with open(csvfile, "r", errors="replace") as f:
        first_line = ""
        for line in f:
            if line.strip():
                first_line = line
                break

    if "->" in first_line:
        return read_mathematica_rule_csv(csvfile)

    return pd.read_csv(csvfile)


def load_jax(csvfile: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csvfile)

    n_col = pick_column(df, ["nphi", "Nphi", "N_phi"])
    s_col = pick_column(df, ["best_S4", "S4", "action", "best_action"])
    t_col = pick_column(df, ["cumulative_total_time_s", "total_time_s", "time_s", "runtime_s"])

    out = pd.DataFrame({
        "nphi": pd.to_numeric(df[n_col], errors="coerce"),
        "action_jax": pd.to_numeric(df[s_col], errors="coerce"),
        "time_jax": pd.to_numeric(df[t_col], errors="coerce"),
    })

    return out.dropna(subset=["nphi"]).sort_values("nphi")


def load_ct(csvfile: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csvfile)

    n_col = pick_column(df, ["nphi", "Nphi", "N_phi"])
    s_col = pick_column(df, ["S4_CT", "action", "S4"])
    t_col = pick_column(df, ["time_s", "runtime_s", "TimeSeconds"])
    status_col = pick_column(df, ["status", "Status"], required=False)

    if status_col is not None:
        df = df[df[status_col].astype(str).str.lower() == "ok"]

    out = pd.DataFrame({
        "nphi": pd.to_numeric(df[n_col], errors="coerce"),
        "action_ct": pd.to_numeric(df[s_col], errors="coerce"),
        "time_ct": pd.to_numeric(df[t_col], errors="coerce"),
    })

    return out.dropna(subset=["nphi"]).sort_values("nphi")


def load_fb(csvfile: str | Path) -> pd.DataFrame:
    df = read_normal_or_rule_csv(csvfile)

    n_col = pick_column(df, ["nphi", "Nphi", "N_phi"])
    s_col = pick_column(df, ["ActionFindBounceD4", "S4_FB", "action", "S4"])
    t_col = pick_column(df, ["TimeSeconds", "time_s", "runtime_s"])
    status_col = pick_column(df, ["status", "Status"], required=False)

    if status_col is not None:
        df = df[df[status_col].astype(str).str.lower() == "ok"]

    out = pd.DataFrame({
        "nphi": df[n_col].apply(to_number),
        "action_fb": df[s_col].apply(to_number),
        "time_fb": df[t_col].apply(to_number),
    })

    return out.dropna(subset=["nphi"]).sort_values("nphi")


def apply_style():
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
        "lines.markersize": 5.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })



def make_series_with_gaps(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """
    Return a full integer-N_phi series with NaN values inserted for missing points.

    This prevents matplotlib from drawing a misleading line across missing data.
    Example: if N_phi=40 and 42 exist but 41 is missing, the curve breaks at 41.
    """
    valid_x = pd.to_numeric(df[xcol], errors="coerce").dropna()
    if valid_x.empty:
        return pd.DataFrame({xcol: [], ycol: []})

    x_min = int(valid_x.min())
    x_max = int(valid_x.max())

    full = pd.DataFrame({xcol: np.arange(x_min, x_max + 1)})

    sub = df[[xcol, ycol]].copy()
    sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
    sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
    sub = sub.dropna(subset=[xcol])
    sub[xcol] = sub[xcol].astype(int)

    # If there are repeated nphi values, keep the first non-null one.
    sub = sub.sort_values(xcol).groupby(xcol, as_index=False)[ycol].first()

    return full.merge(sub, on=xcol, how="left")



def make_action_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    sub = make_series_with_gaps(df, "nphi", "action_jax")
    ax.plot(sub["nphi"], sub["action_jax"], marker="o", label="JAX-Fourier")

    sub = make_series_with_gaps(df, "nphi", "action_ct")
    ax.plot(sub["nphi"], sub["action_ct"], marker="s", label="CosmoTransitions")

    sub = make_series_with_gaps(df, "nphi", "action_fb")
    ax.plot(sub["nphi"], sub["action_fb"], marker="^", label="FindBounce")

    ax.set_xlabel(r"Number of fields $N_\phi$")
    ax.set_ylabel(r"Action $S_4$")
    ax.set_title("Mega-random benchmark: action comparison")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(OUTDIR / "mega_random_action_vs_nphi.png")
    fig.savefig(OUTDIR / "mega_random_action_vs_nphi.pdf")
    plt.close(fig)


def make_time_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    sub = make_series_with_gaps(df, "nphi", "time_jax")
    ax.plot(sub["nphi"], sub["time_jax"], marker="o", label="JAX-Fourier")

    sub = make_series_with_gaps(df, "nphi", "time_ct")
    ax.plot(sub["nphi"], sub["time_ct"], marker="s", label="CosmoTransitions")

    sub = make_series_with_gaps(df, "nphi", "time_fb")
    ax.plot(sub["nphi"], sub["time_fb"], marker="^", label="FindBounce")

    ax.set_xlabel(r"Number of fields $N_\phi$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Mega-random benchmark: runtime comparison")
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(OUTDIR / "mega_random_time_vs_nphi.png")
    fig.savefig(OUTDIR / "mega_random_time_vs_nphi.pdf")
    plt.close(fig)


def main():
    apply_style()

    print("Loading CSV files...")

    jax = load_jax(JAX_CSV)
    print(f"Loaded JAX: {len(jax)} rows")

    ct = load_ct(CT_CSV)
    print(f"Loaded CosmoTransitions: {len(ct)} rows")

    fb = load_fb(FB_CSV)
    print(f"Loaded FindBounce: {len(fb)} rows")

    df = jax.merge(ct, on="nphi", how="outer")
    df = df.merge(fb, on="nphi", how="outer")
    df = df.sort_values("nphi")

    # Make nphi integer-looking where possible.
    df["nphi"] = df["nphi"].astype(int)

    merged_out = OUTDIR / "mega_random_merged_summary.csv"
    df.to_csv(merged_out, index=False)

    print("\nMerged table preview:")
    print(df.head(15).to_string(index=False))

    make_action_plot(df)
    make_time_plot(df)

    print("\nSaved:")
    print("  plots/mega_random_action_vs_nphi.png")
    print("  plots/mega_random_action_vs_nphi.pdf")
    print("  plots/mega_random_time_vs_nphi.png")
    print("  plots/mega_random_time_vs_nphi.pdf")
    print(f"  {merged_out}")


if __name__ == "__main__":
    main()
