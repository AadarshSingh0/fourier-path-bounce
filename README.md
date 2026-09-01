# Fourier Bounce Benchmark Codes

This repository contains the benchmark codes used for the Fourier-deformation study of multi-field bounce solutions. The scripts are organized by task: paper benchmarks, 2D visual checks, Fourier-assisted standard algorithms, and basis-comparison studies.

The original paper reproductions remain available as scripts. A reusable
`fourier_path_bounce` Python package now provides the same fixed-\(V_t\)
Fourier preconditioner plus adapters for CosmoTransitions and FindBounce.

## Reusable preconditioning API

The public convention is always an array of shape `(n_points, n_fields)`, with
the false vacuum exactly at row 0 (`t=0`) and the true vacuum exactly at the
last row (`t=1`). The Fourier coefficients have shape
`(n_fields, selected_modes)`. The current production profile is the published
cubic smoothstep tied to raw `t`; the false-vacuum energy is shifted to zero.

The Fourier stage creates an endpoint-preserving, action-informed initial
field-space path. It is not the final bounce solver. CosmoTransitions performs
its standard path deformation after receiving the sampled path. FindBounce
receives an open polygonal sequence consisting of the false vacuum, `K`
interior points, and the true vacuum.

### Reusable-package documentation

- [Python API and package file map](fourier_path_bounce/README.md)
- [Wolfram / FindBounce adapter](fourier_path_bounce/wolfram/README.md)
- [Runnable examples](examples/README.md)
- [Finite-temperature real-singlet example](examples/xsm/README.md)
- [Focused tests](tests/README.md)

### Testing

After installing the project, run the focused reusable-package tests with
`python3 -m unittest discover -s tests -v`. Run all discoverable
repository tests with `python3 -m unittest discover -v`.

### Python requirements

Install the package from the repository root with `pip install .`, or use
`pip install -e .` for an editable development installation. Add the optional
CosmoTransitions dependency with `pip install ".[cosmotransitions]"` or
`pip install -e ".[cosmotransitions]"`. The reusable optimizer requires Python
3.10+, NumPy, SciPy, and 64-bit JAX. CosmoTransitions and FindBounce are
external solvers and are not bundled by this package.

The CosmoTransitions adapter is currently verified with CosmoTransitions
2.0.7. The package metadata requires `cosmoTransitions>=2.0.7,<3`; earlier
2.0.x releases are not claimed compatible with current Python/NumPy
environments. The Wolfram handoff was tested with Wolfram Engine 13.2 and
FindBounce 1.1.0. The adapter isolates the CosmoTransitions 2.0.7
`pathDeformation.fullTunneling` call in
`fourier_path_bounce/cosmotransitions.py` and maps Euclidean dimension `d` to
CosmoTransitions `alpha=d-1`.

Package metadata requires Python 3.10 or newer. The current release was
clean-install tested on Linux x86_64 with Python 3.12.3; other Python and
operating-system combinations are not yet covered by public continuous
integration.

Potentials passed to `optimize_fourier_path` must accept one JAX array of shape
`(n_fields,)`, return one scalar, and be JAX differentiable. Gradients passed
to solver adapters accept one NumPy point and return shape `(n_fields,)`.

### Effective-potential scope

Loop and thermal corrections enter through the effective potential supplied
by the user; they do not require a change to the Fourier path parametrization.
The potential's field dependence must use differentiable JAX-compatible
operations. Tabulated potentials can be used through a differentiable
interpolation or surrogate, while arbitrary non-differentiable black-box
potential routines are not accepted directly by the current Fourier
optimizer. Constructing, renormalizing, and thermally resumming the physical
effective potential remain the model builder's responsibility.

CosmoTransitions or FindBounce performs the final bounce calculation using
the supplied physical potential. This package is a preconditioner and
initializer, not a replacement for either solver.

```python
from fourier_path_bounce import (
    FourierPathSettings,
    ModeSelectionSettings,
    optimize_fourier_path,
)
from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, potential

settings = FourierPathSettings(
    dimension=4,
    n_grid=260,
    mode_selection=ModeSelectionSettings(modes=(1, 2, 3, 4, 5)),
)
result = optimize_fourier_path(
    potential, FALSE_VACUUM, TRUE_VACUUM, settings=settings
)
print(result.path_points.shape, result.coefficients.shape, result.action_proxy)
```

`OptimizerSettings` controls L-BFGS-B tolerances and coefficient bounds.
`ModeSelectionSettings` controls the deterministic mode schedule, adaptive
stopping, warm starts, random starts, and seed. Results expose coefficients,
selected `N_m`, sampled points, proxy action, per-mode history, optimizer and
adaptive status, gradient norm, timings, bound saturation, and admissibility
diagnostics. `result.sample_at(parameter_values)` supports explicit nonuniform
raw-`t` placement. `save_fourier_result` and `load_fourier_result` provide a checked
NPZ round trip. Invalid dimensions, non-finite endpoints, malformed potential
outputs, endpoint mismatches, and solver failures raise explicit exceptions.

### CosmoTransitions

```python
from fourier_path_bounce import CosmoTransitionsSettings, run_cosmotransitions
from examples.reusable_potential import gradient

ct = run_cosmotransitions(
    potential,
    gradient,
    FALSE_VACUUM,
    TRUE_VACUUM,
    fourier_result=result,
    n_path_points=120,
    settings=CosmoTransitionsSettings(dimension=4, maxiter=40),
)
print(ct.action, ct.deformation_steps, ct.f_ratio, ct.status)
```

The adapter reverses the public false-to-true array only at the
CosmoTransitions boundary because `fullTunneling` 2.0.7 expects true-to-false
points. It records Fourier preprocessing, solver, and total times separately.
Use `initialization="straight"` for a direct comparison. A returned result that
uses every configured outer iteration is labeled
`returned_at_outer_iteration_limit`; inspect `f_ratio` rather than treating a
finite action alone as proof of convergence.

### FindBounce

`N_m` is the number of optimized Fourier modes. `K` is independently the
number of interior points injected into FindBounce. Exporting a different `K`
does not rerun the Fourier optimizer.

```python
from fourier_path_bounce import prepare_findbounce_points, export_findbounce_points

one = prepare_findbounce_points(result, K=1, sampling="parameter")
many = prepare_findbounce_points(result, K=4, sampling="arclength")
export_findbounce_points(many, "findbounce_input", basename="initializer")
```

Parameter sampling is uniform in Fourier `t`; arc-length sampling is uniform
in normalized canonical field-space length. The export contains full-precision
interior and full open-path CSV files plus JSON metadata with endpoints,
orientation, dimensions, `N_m`, `K`, sampling method, and SHA-256 hashes.

The generic Wolfram boundary does not translate Python potential source. The
user defines the potential and fields in Wolfram and passes the geometry:

```wolfram
Get["fourier_path_bounce/wolfram/FourierPathBounce.wl"];
result = FourierPathBounce`RunFindBounceWithFourier[
  potential, fields, falseVacuum, trueVacuum, 4,
  "findbounce_input/initializer_metadata.json",
  "Gradient" -> gradient,
  "FindBounceOptions" -> {"PathTolerance" -> 0.01}
];
```

`ImportFourierPathPoints[source, falseVacuum, trueVacuum]` also accepts an
explicit point matrix or full-path CSV. It rejects incorrect dimensions,
non-finite data, reversed endpoints, duplicated endpoint rows, and attempts to
override `"FieldPoints"`, `"Gradient"`, or `Dimension` through the option list.

### Command line and examples

The CLI loads Python callables only through `module:function` imports; it does
not evaluate command-line code.

```bash
python3 -m fourier_path_bounce --help
python3 -m fourier_path_bounce optimize \
  --potential examples.reusable_potential:potential \
  --false=-0.9456492739235919,-0.03701160775472421 \
  --true=1.0466805318046022,0.033439047480567696 \
  --dimension 4 --modes 1,2,3 --output-dir reusable_output
python3 -m fourier_path_bounce sample \
  --result reusable_output/fourier_result.npz --points 80 \
  --sampling arclength --output reusable_output/path.csv
python3 -m fourier_path_bounce ct-prepare \
  --result reusable_output/fourier_result.npz --points 120 \
  --output-dir reusable_output/ct
python3 -m fourier_path_bounce findbounce-export \
  --result reusable_output/fourier_result.npz --K 4 \
  --sampling arclength --output-dir reusable_output/findbounce
```

Run the complete examples as modules from the repository root:

```bash
python3 -m examples.reusable_api_example
python3 -m examples.reusable_cosmotransitions_example
python3 -m examples.reusable_findbounce_example
wolframscript -file examples/reusable_findbounce_example.wls
```

The reusable finite-temperature xSM model and its documented end-to-end
example are in [`examples/xsm/`](examples/xsm/README.md). The example uses the
same leading high-temperature benchmark as BubbleProfiler and FindBounce and
demonstrates tested handoffs to CosmoTransitions 2.0.7 and FindBounce 1.1.0.
Those external packages remain the final bounce solvers and are not bundled.

The benchmark-specific scripts below are retained unchanged as paper-result
reproductions; the reusable package does not import benchmark potentials,
coefficient tables, filenames, or local paths.

---

## 1. Repository structure

```text
.
├── pyproject.toml
├── LICENSE
├── fourier_path_bounce
│   ├── core.py
│   ├── cosmotransitions.py
│   ├── findbounce.py
│   ├── serialization.py
│   ├── models
│   │   └── xsm.py
│   └── wolfram
│       └── FourierPathBounce.wl
├── examples
│   ├── reusable_api_example.py
│   ├── reusable_cosmotransitions_example.py
│   ├── reusable_findbounce_example.py
│   └── xsm
│       ├── benchmark.py
│       └── README.md
├── tests
│   ├── fixtures
│   └── test_*.py
├── 2D Potential
│   ├── basis_paths_action_convergence.py
│   ├── cosmotransitions_optibounce2d_path.py
│   └── Info.txt
├── Fourier+ALGO
│   ├── CT+Fourier
│   │   ├── fourier_ct_scan_data_generator.py
│   │   ├── hybrid_fourier_ct_fixedpath_benchmark.py
│   │   ├── Info.txt
│   │   └── plot_ct_summary.py
│   └── FindBounce+Fourier
│       ├── findbounce_csvK_scan_READY.wl
│       ├── FindBounce.nb
│       ├── Info.txt
│       └── make_fourier_csv_for_findbounce_csvdata.py
├── Info.txt
├── Paper_Benchmark
│   ├── 3D
│   │   ├── Info.txt
│   │   └── jax_fourier_optibounce_D3_compare_plot_uniform.py
│   ├── 4D
│   │   ├── FindBounce.nb
│   │   ├── Info.txt
│   │   ├── mega_random_coefficients.csv
│   │   ├── mega_random_cosmotransitions_check.py
│   │   ├── mega_random_findbounce_runner_FIXED.wl
│   │   ├── mega_random_jax_fourier.py
│   │   └── plot_mega_random_comparison.py
│   ├── Path_Basis
│   │   ├── basis_path.py
│   │   └── Info.txt
│   └── Various_Basis
│       ├── basis_comparison_jax.py
│       ├── Info_Extended.md
│       ├── Info.txt
│       ├── input_data
│       │   └── mega_random_coefficients.csv
│       ├── plot_basis_comparison_from_csv.py
│       └── plot_saved_basis_paths.py
└── README.md

```

---

## 2. Requirements

### 2.1 Python

Recommended:

```text
Python 3.10 or newer
```

Installing the reusable package with `pip install .` installs its mandatory
JAX, NumPy, and SciPy dependencies. The retained paper-benchmark scripts may
additionally require:

```bash
pip install numpy scipy pandas matplotlib
pip install "jax[cpu]"
pip install PyWavelets
```

For scripts using CosmoTransitions:

```bash
pip install "cosmoTransitions>=2.0.7,<3"
```

Version 2.0.7 is the version verified in the current clean Linux/Python 3.12.3
environment. Earlier 2.0.x releases are not claimed compatible with current
Python/NumPy environments.

### 2.2 Mathematica / Wolfram Language

Some benchmarks use **FindBounce**, which is a Mathematica package. You need:

```text
Wolfram Mathematica / Wolfram Language
FindBounce package or paclet
```

If using a paclet file, install it inside Mathematica, for example:

```mathematica
PacletInstall["FindBounce-1.1.0.paclet"]
Needs["FindBounce`"]
```

A quick check is:

```mathematica
Needs["FindBounce`"]
```

If this runs without error, FindBounce is available.

---

## 3. Main benchmark folders

The repository has five main working areas:

```text
Paper_Benchmark/3D
Paper_Benchmark/4D
Paper_Benchmark/Various_Basis
2D Potential
Fourier+ALGO
```

Each folder is described below.

---

# Part I: Paper benchmarks

---

## 4. Published OptiBounce \(D=3\) benchmark

Folder:

```text
Paper_Benchmark/3D
```

Main file:

```text
jax_fourier_optibounce_D3_compare_plot_uniform.py
```

This script compares the JAX-Fourier method with the published OptiBounce benchmark values for the \(D=3\) action.

Run:

```bash
cd "Paper_Benchmark/3D"
python3 jax_fourier_optibounce_D3_compare_plot_uniform.py
```

This benchmark is used for comparison with the published OptiBounce table. It is the correct benchmark to use when comparing against the literature values because the published table uses \(D=3\).

Typical outputs include CSV files and plots comparing:

```text
JAX-Fourier action
published OptiBounce action
FindBounce action
CosmoTransitions action where available
runtime comparison
```

Important caveat: the \(N_\phi=4\) case can converge to a higher-action local minimum in the default local optimizer. This point should be kept transparently if shown, but excluded from clean accuracy summaries when appropriate.

---

## 5. Mega-random \(D=4\) benchmark

Folder:

```text
Paper_Benchmark/4D
```

Main files:

```text
mega_random_jax_fourier.py
mega_random_cosmotransitions_check.py
mega_random_findbounce_runner_FIXED.wl
plot_mega_random_comparison.py
mega_random_coefficients.csv
FindBounce.nb
```

This benchmark uses the random high-dimensional potential family and compares:

```text
JAX-Fourier
CosmoTransitions
FindBounce
```

### 5.1 Run JAX-Fourier

```bash
cd "Paper_Benchmark/4D"
python3 mega_random_jax_fourier.py
```

This produces:

```text
mega_random_jax_history.csv
mega_random_jax_summary.csv
```

This script has the random coefficients hard-coded, so it does not require `mega_random_coefficients.csv`.

### 5.2 Run CosmoTransitions

```bash
python3 mega_random_cosmotransitions_check.py
```

This requires:

```text
mega_random_coefficients.csv
```

and produces:

```text
mega_random_cosmotransitions_results.csv
```

In the present mega-random benchmark and tested environment, CosmoTransitions
failed for the cases with \(N_\phi>10\). We therefore report CosmoTransitions
comparisons only through \(N_\phi=10\). This is an observed limitation of these
runs and should not be interpreted as a universal field-count bound for all
CosmoTransitions applications.

### 5.3 Run FindBounce

Use either `FindBounce.nb` or load the Wolfram script manually.

Inside the Mathematica notebook, run:

```mathematica
ClearAll["Global`*"];
SetDirectory[NotebookDirectory[]];

Needs["FindBounce`"];

Get["mega_random_findbounce_runner_FIXED.wl"];

results = runAll[Range[1, 50]];

exportResults[results, "mega_random_findbounce_results.csv"];
```

For a quick test:

```mathematica
results = runAll[{1, 2, 3, 4, 5}];
exportResults[results, "mega_random_findbounce_results_test.csv"];
```

### 5.4 Plot all three methods together

After the JAX, CosmoTransitions, and FindBounce outputs exist, run:

```bash
python3 plot_mega_random_comparison.py
```

Expected input files:

```text
mega_random_jax_summary.csv
mega_random_cosmotransitions_results.csv
mega_random_findbounce_results.csv
```

Typical outputs:

```text
plots/mega_random_action_vs_nphi.png
plots/mega_random_action_vs_nphi.pdf
plots/mega_random_time_vs_nphi.png
plots/mega_random_time_vs_nphi.pdf
plots/mega_random_merged_summary.csv
```

The plotting code is designed to leave gaps if an intermediate \(N_\phi\) point is missing.

---

# Part II: Basis comparison

---

## 6. Various endpoint-safe basis benchmark

Folder:

```text
Paper_Benchmark/Various_Basis
```

Main files:

```text
basis_comparison_jax.py
plot_basis_comparison_from_csv.py
plot_saved_basis_paths.py
input_data/mega_random_coefficients.csv
```

This workflow compares different endpoint-safe bases:

```text
fourier
chebyshev
legendre
bspline_local
gaussian_local
hybrid_fourier_gaussian
hybrid_fourier_bspline
wavelet_db2
wavelet_db4
wavelet_db6
wavelet_sym4
wavelet_coif1
```

The path is written as

```text
straight path + endpoint-safe basis deformation
```

and the number of basis functions is chosen adaptively.

---

## 6.1 Main computation

Run from:

```text
Paper_Benchmark/Various_Basis
```

### OptiBounce \(D=3\)

```bash
python3 basis_comparison_jax.py --run-mode optibounce_D3
```

This is the main basis-validation run against the \(D=3\) literature benchmark.

### OptiBounce \(D=4\)

```bash
python3 basis_comparison_jax.py --run-mode optibounce_D4
```

This uses the same OptiBounce potential family but evaluates the \(D=4\) action. Do not directly compare the \(D=4\) action values to the published \(D=3\) table.

### Mega-random \(D=4\)

```bash
python3 basis_comparison_jax.py --run-mode mega_random_D4
```

This requires:

```text
input_data/mega_random_coefficients.csv
```

### Quick test

```bash
python3 basis_comparison_jax.py --run-mode optibounce_D3 \
  --nphi 5,10,20 \
  --basis fourier,bspline_local,hybrid_fourier_bspline \
  --no-npz
```

The `--no-npz` option skips saved path files and only writes CSV results.

---

## 6.2 Outputs of the basis computation

The main script writes:

```text
results/<run-mode>/basis_comparison_summary.csv
results/<run-mode>/basis_comparison_history.csv
results/<run-mode>/<case>__<basis>/mode_*.npz
```

The summary CSV is used for tables and summary plots.

The `.npz` files store optimized path data, for example:

```text
coeffs
basis_matrix
t_grid
path
straight_path
false_vac
true_vac
V_path
action
basis_name
n_basis
nphi
potential_name
case_label
action_dim
```

These are used by `plot_saved_basis_paths.py` to redraw path plots without rerunning the optimizer.

---

## 6.3 Make summary plots

Run:

```bash
python3 plot_basis_comparison_from_csv.py --run-label optibounce_D3 --no-show
```

Other choices:

```bash
python3 plot_basis_comparison_from_csv.py --run-label optibounce_D4 --no-show
python3 plot_basis_comparison_from_csv.py --run-label mega_random_D4 --no-show
```

Outputs are saved under:

```text
plots/<run-label>/
```

Typical plots:

```text
basis_time_vs_nphi.png/pdf
basis_relative_error_vs_nphi.png/pdf
basis_relative_error_good_bases.png/pdf
basis_action_vs_nphi.png/pdf
basis_modes_vs_nphi.png/pdf
```

For OptiBounce plots, the script uses the clean range \(N_\phi=5,\ldots,20\) by default to avoid the known \(N_\phi=4\) local-minimum caveat.

---

## 6.4 Make saved-path plots

Example for \(N_\phi=10\), \(D=3\):

```bash
python3 plot_saved_basis_paths.py --run-mode optibounce_D3 \
  results/optibounce_D3/optibounce_n10_d3__fourier \
  results/optibounce_D3/optibounce_n10_d3__bspline_local \
  results/optibounce_D3/optibounce_n10_d3__hybrid_fourier_bspline
```

This saves plots to:

```text
plots/optibounce_D3/path_plots/
```

For \(D=4\):

```bash
python3 plot_saved_basis_paths.py --run-mode optibounce_D4 \
  results/optibounce_D4/optibounce_n10_d4__fourier \
  results/optibounce_D4/optibounce_n10_d4__bspline_local \
  results/optibounce_D4/optibounce_n10_d4__hybrid_fourier_bspline
```

For mega-random:

```bash
python3 plot_saved_basis_paths.py --run-mode mega_random_D4 \
  results/mega_random_D4/mega_random_n20_d4__fourier \
  results/mega_random_D4/mega_random_n20_d4__bspline_local \
  results/mega_random_D4/mega_random_n20_d4__hybrid_fourier_bspline
```

If a folder name differs, inspect it first with:

```bash
ls results/mega_random_D4
```

---

# Part III: 2D path checks

---

## 7. Two-dimensional potential checks

Folder:

```text
2D Potential
```

Main files:

```text
basis_paths_action_convergence.py
cosmotransitions_optibounce2d_path.py
```

These scripts are used for visual 2D path checks on an OptiBounce-like two-field potential.

### 7.1 Fourier/basis path and convergence

```bash
cd "2D Potential"
python3 basis_paths_action_convergence.py --basis all
```

This produces path and convergence plots for the 2D potential.

### 7.2 CosmoTransitions 2D path

```bash
python3 cosmotransitions_optibounce2d_path.py
```

This produces the CosmoTransitions path for the same 2D potential.

These plots are useful for visualizing how the Fourier-deformed path compares with a CosmoTransitions path in a two-field example.

---

# Part IV: Fourier as an initial guess for standard algorithms

---

## 8. Fourier + CosmoTransitions

Folder:

```text
Fourier+ALGO/CT+Fourier
```

Main files:

```text
fourier_ct_scan_data_generator.py
hybrid_fourier_ct_fixedpath_benchmark.py
plot_ct_summary.py
```

This workflow tests whether a Fourier-deformed path can be used as a better initial path for CosmoTransitions.

### 8.1 Generate CT scan data

```bash
cd "Fourier+ALGO/CT+Fourier"
python3 fourier_ct_scan_data_generator.py
```

This generates the data for straight-path versus Fourier-initialized CosmoTransitions runs.

### 8.2 Fixed-path / hybrid benchmark

```bash
python3 hybrid_fourier_ct_fixedpath_benchmark.py
```

This is another benchmark script for the Fourier+CosmoTransitions hybrid setup.

### 8.3 Plot the CT summary

```bash
python3 plot_ct_summary.py
```

Existing example plots are stored in:

```text
Plots/ct_preconditioning_steps.png
Plots/ct_preconditioning_time.png
```

---

## 9. Fourier + FindBounce

Folder:

```text
Fourier+ALGO/FindBounce+Fourier
```

Main files:

```text
make_fourier_csv_for_findbounce_csvdata.py
findbounce_csvK_scan_READY.wl
FindBounce.nb
```

This workflow tests FindBounce using Fourier-generated initial paths.

### 9.1 Generate Fourier path CSV files

```bash
cd "Fourier+ALGO/FindBounce+Fourier"
python3 make_fourier_csv_for_findbounce_csvdata.py
```

This creates a folder:

```text
csv_data/
```

with files like:

```text
fourier_path_n2.csv
fourier_path_n3.csv
...
fourier_path_n50.csv
```

### 9.2 Run FindBounce with Fourier paths

Open `FindBounce.nb`, or run the Wolfram file manually from a notebook.

In Mathematica:

```mathematica
ClearAll["Global`*"];
SetDirectory[NotebookDirectory[]];

Needs["FindBounce`"];

Get["findbounce_csvK_scan_READY.wl"];
```

Then run the scan commands defined inside the file.

The Mathematica code compares straight-path FindBounce with several Fourier-CSV-injected initial paths.

---

# Part V: Data dependencies

---

## 10. Which scripts need external CSV files?

### Does not need an external potential CSV

```text
Paper_Benchmark/3D/jax_fourier_optibounce_D3_compare_plot_uniform.py
Paper_Benchmark/4D/mega_random_jax_fourier.py
Paper_Benchmark/Various_Basis/basis_comparison_jax.py --run-mode optibounce_D3
Paper_Benchmark/Various_Basis/basis_comparison_jax.py --run-mode optibounce_D4
```

### Needs `mega_random_coefficients.csv`

```text
Paper_Benchmark/4D/mega_random_cosmotransitions_check.py
Paper_Benchmark/Various_Basis/basis_comparison_jax.py --run-mode mega_random_D4
```

Required files:

```text
Paper_Benchmark/4D/mega_random_coefficients.csv
Paper_Benchmark/Various_Basis/input_data/mega_random_coefficients.csv
```

### Needs generated Fourier path CSVs

```text
Fourier+ALGO/FindBounce+Fourier/findbounce_csvK_scan_READY.wl
```

Generate them first using:

```text
Fourier+ALGO/FindBounce+Fourier/make_fourier_csv_for_findbounce_csvdata.py
```

---

# Part VI: Known caveats

---

## 11. Known caveats

### \(N_\phi=4\) OptiBounce caveat

The \(N_\phi=4\) OptiBounce case can converge to a higher-action local minimum in the default local optimizer. This is an optimizer-basin issue, not a failure of the basis or Fourier idea.

For clean basis-comparison plots, use:

```text
N_phi = 5,...,20
```

The basis plotting script does this automatically for OptiBounce runs.

### \(D=3\) and \(D=4\) should not be mixed

The published OptiBounce benchmark uses \(D=3\). 
The \(D=4\) action is a different action and should not be directly compared with the \(D=3\) table.

### Benchmark-specific CosmoTransitions failures above \(N_\phi=10\)

In the present mega-random benchmark and tested environment, CosmoTransitions
failed for the cases with \(N_\phi>10\). We therefore report CosmoTransitions
comparisons only through \(N_\phi=10\). This is an observed limitation of these
runs and should not be interpreted as a universal field-count bound for all
CosmoTransitions applications.

### Mathematica working directory

When running Mathematica notebooks, use:

```mathematica
SetDirectory[NotebookDirectory[]];
```

This makes input/output files relative to the notebook folder.

### `.npz` files

The `.npz` files store optimized paths. They can take disk space. For quick basis tests, use:

```bash
--no-npz
```

with `basis_comparison_jax.py`.

---

# Part VII: Minimal reproduction commands

---

## 12. Minimal reproduction

### 12.1 Published \(D=3\) benchmark

```bash
cd "Paper_Benchmark/3D"
python3 jax_fourier_optibounce_D3_compare_plot_uniform.py
```

### 12.2 Mega-random \(D=4\) benchmark

```bash
cd "Paper_Benchmark/4D"

python3 mega_random_jax_fourier.py
python3 mega_random_cosmotransitions_check.py
```

Then run the FindBounce notebook commands described above, and finally:

```bash
python3 plot_mega_random_comparison.py
```

### 12.3 Basis-comparison benchmark

```bash
cd "Paper_Benchmark/Various_Basis"

python3 basis_comparison_jax.py --run-mode optibounce_D3

python3 plot_basis_comparison_from_csv.py --run-label optibounce_D3 --no-show

python3 plot_saved_basis_paths.py --run-mode optibounce_D3 \
  results/optibounce_D3/optibounce_n10_d3__fourier \
  results/optibounce_D3/optibounce_n10_d3__bspline_local \
  results/optibounce_D3/optibounce_n10_d3__hybrid_fourier_bspline
```

### 12.4 2D path checks

```bash
cd "2D Potential"

python3 basis_paths_action_convergence.py --basis all
python3 cosmotransitions_optibounce2d_path.py
```

### 12.5 Fourier + CosmoTransitions

```bash
cd "Fourier+ALGO/CT+Fourier"

python3 fourier_ct_scan_data_generator.py
python3 plot_ct_summary.py
```

### 12.6 Fourier + FindBounce

```bash
cd "Fourier+ALGO/FindBounce+Fourier"

python3 make_fourier_csv_for_findbounce_csvdata.py
```

Then open `FindBounce.nb` and run the Mathematica scan.

---

## 13. External tools to cite

This repository uses or compares against:

```text
JAX
SciPy
CosmoTransitions
FindBounce
OptiBounce benchmark data
```

If using these results in a paper, please cite the relevant original papers/tools.

---

## 14. Final note

The repository is meant to reproduce the numerical tables and figures for the Fourier-deformation bounce project. The codes are explicit and folder-based so that each benchmark can be rerun independently.

## 15. License

This software is released under the [MIT License](LICENSE).
