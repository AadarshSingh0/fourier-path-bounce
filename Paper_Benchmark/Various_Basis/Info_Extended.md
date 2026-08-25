# Various Basis Benchmark

This folder contains the final workflow for comparing different endpoint-safe basis functions in the Fourier bounce project.

The workflow has three Python files:

```text
basis_comparison_jax.py            -> runs the JAX basis comparison
plot_basis_comparison_from_csv.py  -> makes summary plots from the CSV file
plot_saved_basis_paths.py          -> makes path plots from saved .npz files
```

The file `Info.txt` can be used for short notes about the folder.

The basic workflow is:

```text
run optimizer  ->  make summary plots  ->  make saved-path plots
```

---

## 1. Folder layout

A clean folder should look like this:

```text
Various_Basis/
  basis_comparison_jax.py
  plot_basis_comparison_from_csv.py
  plot_saved_basis_paths.py
  Info.txt

  input_data/
    mega_random_coefficients.csv

  results/
    optibounce_D3/
    optibounce_D4/
    mega_random_D4/

  plots/
    optibounce_D3/
    optibounce_D4/
    mega_random_D4/
```

The `results/` and `plots/` folders are created automatically by the scripts.

---

## 2. Main computation file

The main file is:

```text
basis_comparison_jax.py
```

It runs the JAX optimizer and compares different basis choices.

It saves:

```text
results/<run-mode>/basis_comparison_summary.csv
results/<run-mode>/basis_comparison_history.csv
results/<run-mode>/<case>__<basis>/mode_*.npz
```

The `.csv` files are used for tables and summary plots.  
The `.npz` files store the optimized paths and are used later by `plot_saved_basis_paths.py`.

---

## 3. How to run the main computation

### OptiBounce benchmark, \(D=3\)

Use this for comparison with the published OptiBounce table.

```bash
python3 basis_comparison_jax.py --run-mode optibounce_D3
```

### OptiBounce benchmark, \(D=4\)

Same potential family, but using the \(D=4\) action.

```bash
python3 basis_comparison_jax.py --run-mode optibounce_D4
```

Do not directly compare these \(D=4\) numbers with the published \(D=3\) OptiBounce table.

### Mega-random benchmark, \(D=4\)

```bash
python3 basis_comparison_jax.py --run-mode mega_random_D4
```

For this run, the file below should be present:

```text
input_data/mega_random_coefficients.csv
```

---

## 4. Quick test run

For a fast check, run only a few cases and bases:

```bash
python3 basis_comparison_jax.py --run-mode optibounce_D3 \
  --nphi 5,10,20 \
  --basis fourier,bspline_local,hybrid_fourier_bspline \
  --no-npz
```

The option `--no-npz` skips the saved path files and keeps only the CSV outputs.

---

## 5. Useful options for the main code

```text
--run-mode optibounce_D3
--run-mode optibounce_D4
--run-mode mega_random_D4

--nphi 5,10,20
--nphi 5:20

--basis fourier,bspline_local,hybrid_fourier_bspline

--no-npz
--tolerance 1e-3
--patience 3
--n-basis-max 25
--n-grid 260
```

---

## 6. Output files from the main code

### 6.1 `basis_comparison_summary.csv`

This is the main output file. Use it for tables and plots.

Important columns:

```text
case_label
potential_name
action_dim
nphi
basis
converged
best_n_basis
best_S
reference_FB
rel_to_FB_percent
cumulative_compile_time_s
cumulative_solve_time_s
cumulative_total_time_s
output_dir
```

For `optibounce_D3`, the reference FindBounce values are available, so `rel_to_FB_percent` is meaningful.

For `optibounce_D4`, the reference columns may be empty because the published reference table is for \(D=3\), not \(D=4\).

### 6.2 `basis_comparison_history.csv`

This stores the scan history. It has one row per basis size.

Use it to study convergence with increasing basis size.

Important columns:

```text
case_label
basis
n_basis
mode_best_S
global_best_S_after_mode
global_best_n_basis_after_mode
improved_global_best
no_improve_count
compile_time_s
solve_time_s_sum
cumulative_total_time_s
```

### 6.3 `mode_*.npz`

The `.npz` files store saved path data.

Typical contents:

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

They are useful because they let us redraw paths later without rerunning the optimizer.

---

## 7. Summary plotting file

The summary plotting file is:

```text
plot_basis_comparison_from_csv.py
```

It reads:

```text
results/<run-label>/basis_comparison_summary.csv
```

and saves plots into:

```text
plots/<run-label>/
```

It does not rerun the optimizer.

### Run for OptiBounce \(D=3\)

```bash
python3 plot_basis_comparison_from_csv.py --run-label optibounce_D3
```

### Run for OptiBounce \(D=4\)

```bash
python3 plot_basis_comparison_from_csv.py --run-label optibounce_D4
```

### Run for mega-random \(D=4\)

```bash
python3 plot_basis_comparison_from_csv.py --run-label mega_random_D4
```

### Run without opening plot windows

```bash
python3 plot_basis_comparison_from_csv.py --run-label optibounce_D3 --no-show
```

### Plots created

```text
plots/<run-label>/basis_time_vs_nphi.png
plots/<run-label>/basis_time_vs_nphi.pdf

plots/<run-label>/basis_relative_error_vs_nphi.png
plots/<run-label>/basis_relative_error_vs_nphi.pdf

plots/<run-label>/basis_relative_error_good_bases.png
plots/<run-label>/basis_relative_error_good_bases.pdf

plots/<run-label>/basis_action_vs_nphi.png
plots/<run-label>/basis_action_vs_nphi.pdf

plots/<run-label>/basis_modes_vs_nphi.png
plots/<run-label>/basis_modes_vs_nphi.pdf
```

For OptiBounce plots, the script uses the clean range \(N_\phi=5,\ldots,20\) by default. This avoids the known \(N_\phi=4\) local-minimum caveat.

---

## 8. Saved-path plotting file

The saved-path plotting file is:

```text
plot_saved_basis_paths.py
```

It reads the `.npz` files inside the result folders.  
It does not recompute actions.

It saves path plots into:

```text
plots/<run-mode>/path_plots/
```

### 8.1 Plot mode evolution for one basis

Example: Fourier path evolution for \(N_\phi=10\), \(D=3\):

```bash
python3 plot_saved_basis_paths.py --run-mode optibounce_D3 \
  results/optibounce_D3/optibounce_n10_d3__fourier
```

### 8.2 Compare final paths for several bases

This is the most useful command for the paper basis section:

```bash
python3 plot_saved_basis_paths.py --run-mode optibounce_D3 \
  results/optibounce_D3/optibounce_n10_d3__fourier \
  results/optibounce_D3/optibounce_n10_d3__bspline_local \
  results/optibounce_D3/optibounce_n10_d3__hybrid_fourier_bspline
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

If the folder name is different, check it using:

```bash
ls results/mega_random_D4
```

and copy the exact folder names.

---

## 9. Recommended paper workflow

For the main basis comparison, run:

```bash
python3 basis_comparison_jax.py --run-mode optibounce_D3

python3 plot_basis_comparison_from_csv.py --run-label optibounce_D3 --no-show

python3 plot_saved_basis_paths.py --run-mode optibounce_D3 \
  results/optibounce_D3/optibounce_n10_d3__fourier \
  results/optibounce_D3/optibounce_n10_d3__bspline_local \
  results/optibounce_D3/optibounce_n10_d3__hybrid_fourier_bspline
```

This gives:

```text
1. basis action/error/time plots
2. basis mode-count plots
3. path comparison plots
```

Recommended table choices:

```text
N_phi = 3, 10, 20
Bases = Fourier, Chebyshev, B-spline local, Fourier+B-spline, Wavelet db2
```

---

## 10. Interpretation guide

### Fourier

Good default basis. It is stable and usually gives sub-percent agreement.

### B-spline local

Very competitive. It can capture local path bending efficiently.

### Hybrid Fourier+B-spline

Strong all-rounder. It combines global Fourier flexibility with local deformation.

### Hybrid Fourier+Gaussian

Often good, but usually slower than Fourier+B-spline.

### Chebyshev and Legendre

These global polynomial bases are less effective in the present benchmark.

### Gaussian local

Can be useful in some low-dimensional tests, but it is not the main production basis here.

### Wavelet-inspired bases

Fast, but in the smooth OptiBounce benchmark they give substantially higher actions. They are kept as diagnostic bases, not as main production choices.

---

## 11. Known caveats

### \(N_\phi=4\) OptiBounce caveat

The \(N_\phi=4\) OptiBounce case has a known local-minimum issue in the default local optimizer. This is an optimizer-basin issue, not a failure of the basis idea.

For clean basis-comparison plots, use:

```text
N_phi = 5,...,20
```

The plotting script does this automatically for OptiBounce runs.

### \(D=3\) and \(D=4\) should not be mixed

The published OptiBounce table uses \(D=3\).  
The \(D=4\) action is a different action. Do not directly compare \(D=3\) and \(D=4\) numbers.

### `.npz` files can take space

The `.npz` files are useful for path plots. For quick tests, skip them using:

```bash
--no-npz
```

---

## 12. Minimal commands

Run these from this folder:

```bash
python3 basis_comparison_jax.py --run-mode optibounce_D3

python3 plot_basis_comparison_from_csv.py --run-label optibounce_D3 --no-show

python3 plot_saved_basis_paths.py --run-mode optibounce_D3 \
  results/optibounce_D3/optibounce_n10_d3__fourier \
  results/optibounce_D3/optibounce_n10_d3__bspline_local \
  results/optibounce_D3/optibounce_n10_d3__hybrid_fourier_bspline
```

That is the main basis-comparison workflow.
