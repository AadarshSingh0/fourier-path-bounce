# Reusable examples

This folder shows how to use the public `fourier_path_bounce` API with a
user-supplied potential and how to pass the resulting geometry to external
bounce solvers. Run the examples as modules from the repository root.

## File map

### Public examples

- `__init__.py` marks this directory as an importable Python package so the
  examples can share potential definitions.
- `reusable_potential.py` defines the small two-field potential, gradient,
  false vacuum, and true vacuum used by the public examples.
- `curved_valley_potential.py` defines the established two-field
  curved-valley case used by the CosmoTransitions validation driver.
- `reusable_api_example.py` optimizes and serializes a Fourier path using
  only the solver-independent public API.
- `reusable_cosmotransitions_example.py` optimizes a Fourier initializer,
  runs standard CosmoTransitions path deformation, and prints its result
  summary.
- `reusable_findbounce_example.py` exports arc-length-sampled open paths
  with `K=1` and `K=4` interior points for FindBounce.
- `reusable_findbounce_example.wls` defines the same potential in Wolfram,
  imports the `K=4` Python export, and runs FindBounce twice: once with
  `MaxPathIterations -> 10`, which is not a sufficient budget for this case and
  is reported as `returned_at_path_iteration_limit`, and once with
  `MaxPathIterations -> 20`, which stops on the solver's own tolerance after 11
  iterations. The two actions differ by about 1%, which is why the example
  prints the full termination diagnostics rather than the action alone.
- `xsm/` contains the finite-temperature Z2-symmetric real-singlet Standard
  Model benchmark used in BubbleProfiler and FindBounce. Its documented driver
  validates the model and runs the reusable Fourier, CosmoTransitions, and
  genuine Wolfram/FindBounce workflow; see
  [`xsm/README.md`](xsm/README.md).

### Integration and validation drivers

- `run_reusable_integrations.py` writes auditable straight and
  Fourier-initialized CosmoTransitions results plus `K=1` and `K=4`
  FindBounce geometry exports to a requested output directory.
- `run_cosmotransitions_integration.py` compares straight and Fourier
  initializers on the established curved-valley case and saves paths, logs, and
  a JSON summary.
- `run_reusable_findbounce.wls` is the noninteractive FindBounce validation
  driver. It reads metadata and result paths from the
  `FOURIER_PATH_METADATA` and `FOURIER_PATH_RESULT` environment
  variables.

The integration drivers are retained for repeatable validation. New users
normally start with the public examples.

## Reading FindBounce results

Never quote a FindBounce action without checking how the run terminated. Every
result from `RunFindBounceWithFourier` reports `Status`, `PathIterations`,
`ConfiguredMaxPathIterations`, `PathIterationLimitReached`, `PathConvergence`
and `ConvergenceEvidence`. A run that stops at the configured limit may have
been truncated; a run that stops below it has only satisfied FindBounce's own
`PathTolerance`/`ActionTolerance` rule, which is not a proof of numerical
convergence. Actions obtained with different `K` are not expected to agree, and
these examples make no K-independence or universal-convergence claim. See
[`fourier_path_bounce/wolfram/README.md`](../fourier_path_bounce/wolfram/README.md).

## Prerequisites

Use Python 3.10 or newer and install the project from the repository root. The
base API needs JAX, NumPy, and SciPy. CosmoTransitions support is optional:

```bash
pip install -e .
pip install -e ".[cosmotransitions]"
```

The Wolfram example additionally requires Mathematica or Wolfram Engine and an
installed FindBounce package. CosmoTransitions and FindBounce are external
final bounce solvers; neither solver is bundled with this project.

## Recommended order

From the repository root, run:

```bash
python3 -m examples.reusable_api_example
python3 -m examples.reusable_cosmotransitions_example
python3 -m examples.reusable_findbounce_example
wolframscript -file examples/reusable_findbounce_example.wls
python3 -m examples.xsm.benchmark validate --output-dir xsm_example_output
```

The API example creates
`example_output/fourier_result.npz`. The Python FindBounce example creates
`example_output/findbounce_K1/` and
`example_output/findbounce_K4/`, each containing full-precision interior
points, the complete open path, and metadata. It exports geometry only; the
subsequent Wolfram example consumes the generated `K=4` metadata/path and
performs the final FindBounce calculation. The CosmoTransitions example prints
a structured result summary and does not create an output directory.
