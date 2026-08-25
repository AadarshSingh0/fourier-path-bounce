# `fourier_path_bounce` API

This package extracts the repository's validated fixed-`V_t` JAX-Fourier path
optimization into reusable functions. Public paths are always
false-vacuum-to-true-vacuum arrays with shape `(n_points, n_fields)`.

Install from the repository root with `pip install .`, or use `pip install -e .`
for an editable checkout. The optional `.[cosmotransitions]` extra installs the
supported CosmoTransitions 2.x Python dependency. CosmoTransitions and
FindBounce are external final bounce solvers; they are not bundled with this
package. The adapter was tested with CosmoTransitions 2.0.2, while the Wolfram
boundary was tested with Wolfram Engine 13.2 and FindBounce 1.1.0.

## Public functions

```python
optimize_fourier_path(potential, false_vacuum, true_vacuum, *,
                      settings=None, potential_gradient=None,
                      initial_coefficients=None, metadata=None)
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

`optimize_fourier_path` supports the repository's validated `d=3` and `d=4`
discrete actions. The `d=3` objective retains the published `1e-300` guards
and `1e50` invalid-interval penalty. The `d=4` objective retains the published
unclipped arithmetic. No adapter changes these objectives.

CosmoTransitions 2.x expects its path in true-to-false order. Only
`prepare_cosmotransitions_path` and `run_cosmotransitions` perform that
boundary reversal. `CosmoTransitionsSettings.dimension` is mapped to the
single-field solver's `alpha=dimension-1` and conflicting overrides are
rejected.

The Wolfram package `wolfram/FourierPathBounce.wl` exports:

```wolfram
ImportFourierPathPoints[source, falseVacuum, trueVacuum]
RunFindBounceWithFourier[potential, fields, falseVacuum, trueVacuum,
                         dimension, source, options]
```

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
- `cosmotransitions.py` isolates CosmoTransitions 2.x path conversion and
  `fullTunneling` calls while keeping preprocessing and solver diagnostics
  separate.
- `findbounce.py` samples `K` interior points independently of
  `N_m` and writes or reloads full-precision open-path CSV/JSON exports.
- `wolfram/FourierPathBounce.wl` validates exported geometry and provides
  the generic FindBounce wrapper.
- `wolfram/README.md` documents the Python-to-Wolfram handoff and its
  external requirements.

Runnable examples and their outputs are described in
[`examples/README.md`](../examples/README.md). Focused and repository-wide
test instructions are in [`tests/README.md`](../tests/README.md).
