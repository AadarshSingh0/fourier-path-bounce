#!/usr/bin/env python3
"""
plot_saved_basis_paths.py

Plot paths saved by basis_comparison_jax.py.

This version saves all figures into:
    plots/<RUN_MODE>/path_plots/

Usage examples
--------------
Single basis: show how the optimized path changes as basis size increases

    python3 plot_saved_basis_paths.py \
      results/optibounce_D3/optibounce_n5_d3__fourier

Compare final paths from several bases

    python3 plot_saved_basis_paths.py \
      results/optibounce_D3/optibounce_n10_d3__fourier \
      results/optibounce_D3/optibounce_n10_d3__bspline_local \
      results/optibounce_D3/optibounce_n10_d3__hybrid_fourier_bspline

Optional explicit run mode

    python3 plot_saved_basis_paths.py --run-mode optibounce_D3 \
      results/optibounce_D3/optibounce_n10_d3__fourier \
      results/optibounce_D3/optibounce_n10_d3__bspline_local
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def infer_run_mode_from_folder(folder: Path) -> str:
    """
    Infer RUN_MODE from a path like:
        results/optibounce_D3/optibounce_n10_d3__fourier
    returning:
        optibounce_D3
    """
    parts = folder.parts
    if "results" in parts:
        i = parts.index("results")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "manual_path_plots"


def make_plot_dir(run_mode: str) -> Path:
    plot_dir = Path("plots") / run_mode / "path_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def latest_mode_file(folder: Path) -> Path:
    files = sorted(folder.glob("mode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No mode_*.npz files found in {folder}")
    return files[-1]


def project_path(path: np.ndarray):
    """
    Project an arbitrary-dimensional path to 2D.

    x-axis: direction from false vacuum to true vacuum.
    y-axis: largest perpendicular deformation direction, extracted with SVD.
    """
    false = path[0]
    true = path[-1]

    e1 = true - false
    nrm = np.linalg.norm(e1)
    if nrm < 1e-14:
        e1 = np.zeros_like(e1)
        e1[0] = 1.0
    else:
        e1 = e1 / nrm

    centered = path - false[None, :]
    x = centered @ e1

    straight = false[None, :] + np.linspace(0, 1, len(path))[:, None] * (true - false)[None, :]
    perp = path - straight
    perp = perp - (perp @ e1)[:, None] * e1[None, :]

    if path.shape[1] == 1 or np.linalg.norm(perp) < 1e-12:
        y = np.zeros_like(x)
    else:
        _, _, vh = np.linalg.svd(perp, full_matrices=False)
        y = centered @ vh[0]

    return x, y


def plot_folder_modes(folder: Path, plot_dir: Path):
    """
    For one folder/case/basis, plot how the path changes as basis size increases.
    """
    folder = Path(folder)
    files = sorted(folder.glob("mode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No mode_*.npz files found in {folder}")

    # Plot at most 8 representative mode files to avoid clutter.
    if len(files) > 8:
        idx = np.linspace(0, len(files) - 1, 8, dtype=int)
        files_to_plot = [files[i] for i in idx]
    else:
        files_to_plot = files

    # 1. Projected path plot
    plt.figure(figsize=(7, 5))
    for f in files_to_plot:
        d = load_npz(f)
        x, y = project_path(d["path"])
        plt.plot(x, y, label=f"N={int(d['n_basis'])}")

    plt.xlabel("projection along false-true direction")
    plt.ylabel("largest perpendicular projection")
    plt.title(folder.name)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = plot_dir / f"{folder.name}_projected_modes.png"
    plt.savefig(out, dpi=220)
    print("Saved", out)

    # 2. Potential along path plot
    plt.figure(figsize=(7, 5))
    for f in files_to_plot:
        d = load_npz(f)
        plt.plot(d["t_grid"], d["V_path"], label=f"N={int(d['n_basis'])}")

    plt.xlabel("t")
    plt.ylabel(r"$V(\phi(t))-V_{\rm false}$")
    plt.title(folder.name + " potential along path")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = plot_dir / f"{folder.name}_Vpath_modes.png"
    plt.savefig(out, dpi=220)
    print("Saved", out)


def compare_final_paths(folders, plot_dir: Path):
    """
    Compare the final/latest optimized paths from several basis folders.
    """
    folders = [Path(f) for f in folders]

    # Create informative suffix:
    # optibounce_n10_d3__fourier -> fourier
    names = "__".join([folder.name.split("__")[-1] for folder in folders])

    # 1. Final projected paths
    plt.figure(figsize=(7, 5))
    for folder in folders:
        d = load_npz(latest_mode_file(folder))
        x, y = project_path(d["path"])

        basis = str(d["basis_name"])
        n_basis = int(d["n_basis"])
        action = float(d["action"])

        label = f"{basis}, N={n_basis}, S={action:.4g}"
        plt.plot(x, y, label=label)

    plt.xlabel("projection along false-true direction")
    plt.ylabel("largest perpendicular projection")
    plt.title("Final optimized paths")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = plot_dir / f"compare_final_projected_paths__{names}.png"
    plt.savefig(out, dpi=220)
    print("Saved", out)

    # 2. Final potential along paths
    plt.figure(figsize=(7, 5))
    for folder in folders:
        d = load_npz(latest_mode_file(folder))

        basis = str(d["basis_name"])
        n_basis = int(d["n_basis"])
        action = float(d["action"])

        label = f"{basis}, N={n_basis}, S={action:.4g}"
        plt.plot(d["t_grid"], d["V_path"], label=label)

    plt.xlabel("t")
    plt.ylabel(r"$V(\phi(t))-V_{\rm false}$")
    plt.title("Potential along final paths")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = plot_dir / f"compare_final_Vpaths__{names}.png"
    plt.savefig(out, dpi=220)
    print("Saved", out)


def main():
    parser = argparse.ArgumentParser(description="Plot saved path data from basis_comparison_jax.py")
    parser.add_argument(
        "folders",
        nargs="+",
        help="One or more results folders containing mode_*.npz files.",
    )
    parser.add_argument(
        "--run-mode",
        default=None,
        help=(
            "Optional run mode for plot output folder, e.g. optibounce_D3, "
            "mega_random_D4, kink2d_D4. If omitted, inferred from results/<RUN_MODE>/..."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures but do not open interactive windows.",
    )
    args = parser.parse_args()

    folders = [Path(f) for f in args.folders]

    run_mode = args.run_mode
    if run_mode is None:
        run_mode = infer_run_mode_from_folder(folders[0])

    plot_dir = make_plot_dir(run_mode)
    print(f"Saving path plots to: {plot_dir}")

    if len(folders) == 1:
        plot_folder_modes(folders[0], plot_dir)
    else:
        compare_final_paths(folders, plot_dir)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
