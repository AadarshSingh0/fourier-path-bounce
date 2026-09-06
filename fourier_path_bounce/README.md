# `fourier_path_bounce` API

This package extracts the repository's validated fixed-`V_t` JAX-Fourier path
optimization into reusable functions. Public paths are always
false-vacuum-to-true-vacuum arrays with shape `(n_points, n_fields)`.

Install from the repository root with `pip install .`, or use `pip install -e .`
for an editable checkout. The optional `.[cosmotransitions]` extra installs the
verified `cosmoTransitions>=2.0.7,<3` Python dependency. CosmoTransitions and
FindBounce are external final bounce solvers; they are not bundled with this
package. The current clean environment used CosmoTransitions 2.0.7; earlier
2.0.x releases are not claimed compatible with current Python/NumPy
environments. The Wolfram boundary was tested with Wolfram Engine 13.2 and
FindBounce 1.1.0.

The base installation requires Python 3.10 or newer and installs JAX, NumPy,
and SciPy. The current release was clean-install tested on Linux x86_64 with
Python 3.12.3; other Python and operating-system combinations are not yet
covered by public continuous integration.

## Effective-potential scope

Loop and thermal corrections enter through the user-supplied effective
potential and require no change to the Fourier path parametrization. Its field
dependence must use differentiable JAX-compatible operations. Tabulated
potentials can be used through a differentiable interpolation or surrogate;
arbitrary non-differentiable black-box routines are not accepted directly by
the current optimizer. Constructing, renormalizing, and thermally resumming
the physical effective potential remain the model builder's responsibility.
CosmoTransitions or FindBounce then performs the final bounce calculation
using that physical potential; this package supplies a preconditioner or
initializer and does not replace either solver.

## Public functions

```python
optimize_fourier_path(potential, false_vacuum, true_vacuum, *,
                      settings=None, potential_gradient=None,
                      initial_coefficients=None, metadata=None)
# initial_coefficients is a start strategy in its own right: when it is
# supplied, ModeSelectionSettings may disable zero_start, warm_previous,
# warm_best and random_starts. Without it at least one of those is required.
save_fourier_result(result, path)
load_fourier_result(path)
prepare_cosmotransitions_path(false_vacuum, true_vacuum, *,
                              fourier_result=None, initialization="fourier",
                              n_points=120, sampling="parameter")
run_cosmotransitions(potential, gradient, false_vacuum, true_vacuum, *,
                      fourier_result=None, fourier_settings=None,
                      initialization="fourier", n_path_points=120,
                      sampling="parameter", settings=None)
prepare_findbounce_points(result, K, *, sampling="parameter")
export_findbounce_points(point_set, output_directory, *,
                         basename="findbounce_fourier")
load_findbounce_points(metadata_file)
```

Configuration and result types are exported from `fourier_path_bounce`:

```text
OptimizerSettings
ModeSelectionSettings
FourierPathSettings
FourierPathResult
CosmoTransitionsSettings
CosmoTransitionsResult
FindBouncePointSet
```

`N_m` is the selected Fourier mode count. `K` is the independently selected
number of FindBounce interior points. FindBounce exports are open polygonal
paths, not closed polygons.

### Optimizer starts and the D=3 zero start

The bounded L-BFGS-B optimizer can terminate on its own `ftol` test at its start
point, reporting `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH` after a
single iteration without having moved. On the `d=3` objective this happens
routinely when the only start is the straight line (`zero_start` with
`random_starts=0`, the library default): the returned coefficients are all zero,
so the "preconditioned" path *is* the straight line, even though the straight
path is not stationary there.

This is a real limitation of the bounded optimizer on this objective, not a
property of the potential, and it is now reported rather than hidden. When no
start at any mode count leaves its initial point while the projected gradient
stays above `gtol`, the result carries

```text
optimizer_made_no_progress = True
optimizer_success          = False
adaptive_converged         = False
stop_reason                = "optimizer made no progress from any start ..."
```

**What to configure.** For `d=3`, set `ModeSelectionSettings.random_starts >= 1`
(the shipped examples use `random_starts=3`), or supply explicit
`initial_coefficients`. Outcomes remain seed-dependent, so treat `seed` as a
setting to vary, and always check `optimizer_made_no_progress`,
`gradient_norm`, and `adaptive_converged` before using a path. `d=4` is not
usually affected at the default settings.

`optimize_fourier_path` supports the repository's validated `d=3` and `d=4`
discrete actions. The `d=3` objective retains the published `1e-300` guards
and `1e50` invalid-interval penalty. The `d=4` objective retains the published
unclipped arithmetic. No adapter changes these objectives.

CosmoTransitions 2.0.7 expects its path in true-to-false order. Only
`prepare_cosmotransitions_path` and `run_cosmotransitions` perform that
boundary reversal. `CosmoTransitionsSettings.dimension` is mapped to the
single-field solver's `alpha=dimension-1` and conflicting overrides are
rejected.

The Wolfram package `wolfram/FourierPathBounce.wl` exports:

```wolfram
ImportFourierPathPoints[source, falseVacuum, trueVacuum]
RunFindBounceWithFourier[potential, fields, falseVacuum, trueVacuum,
                         dimension, source, options]
FindBounceTerminationReport[bounce, findBounceOptions, fieldCount]
```

FindBounce results carry explicit termination diagnostics: a finite action is
not treated as convergence, a run that stops at `"MaxPathIterations"` is
reported as `returned_at_path_iteration_limit`, and a run that stops below the
cap is reported as having satisfied the solver's own tolerance rather than as
proven converged. See [`wolfram/README.md`](wolfram/README.md).

Python exports geometry and metadata; users define their potential and
gradient in Wolfram. This avoids unreliable source translation.

## Package file map

- `__init__.py` exports the stable public settings, result types,
  optimization functions, adapters, and serialization helpers.
- `__main__.py` enables `python3 -m fourier_path_bounce` and delegates
  to the command-line interface.
- `cli.py` implements validated optimize, sample, CosmoTransitions
  preparation/run, and FindBounce export commands using
  `module:function` callable imports.
- `core.py` contains endpoint-preserving Fourier path construction, the
  validated D=3 and D=4 proxy actions, adaptive mode selection, and structured
  optimization results.
- `serialization.py` saves and loads checked NPZ/JSON results with shape,
  orientation, endpoint, and checksum validation.
- `cosmotransitions.py` isolates CosmoTransitions 2.0.7 path conversion and
  `fullTunneling` calls while keeping preprocessing and solver diagnostics
  separate.
- `findbounce.py` samples `K` interior points independently of
  `N_m` and writes or reloads full-precision open-path CSV/JSON exports.
- `models/xsm.py` provides the validated leading-high-temperature real-singlet
  benchmark potential, gradient, Hessian, and analytic vacua for NumPy, JAX,
  CosmoTransitions, and the Wolfram handoff; see `models/README.md`.
- `wolfram/FourierPathBounce.wl` validates exported geometry and provides
  the generic FindBounce wrapper.
- `wolfram/README.md` documents the Python-to-Wolfram handoff and its
  external requirements.

Runnable examples and their outputs are described in
[`examples/README.md`](../examples/README.md). Focused and repository-wide
test instructions are in [`tests/README.md`](../tests/README.md).

## Finite-temperature xSM model

`fourier_path_bounce.models.XSMThermalModel` implements the Z2-symmetric
real-singlet Standard Model benchmark used in the BubbleProfiler
(arXiv:1901.03714) and FindBounce (arXiv:2002.00881) studies. It supplies the
thermal potential, analytic derivatives, vacua, and validation utilities
without importing either optional bounce solver.

This is the published leading high-temperature approximation, not a full
one-loop or daisy-resummed precision potential. See the
[xSM example](../examples/xsm/README.md) for the Fourier, CosmoTransitions, and
Wolfram/FindBounce workflow.
