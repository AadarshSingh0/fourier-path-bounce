# Finite-temperature real-singlet benchmark

This example applies the reusable Fourier path preconditioner to the
finite-temperature, Z2-symmetric real-singlet extension of the Standard Model
used by BubbleProfiler and FindBounce. It is the leading high-temperature
approximation from those studies, not a full one-loop or daisy-resummed
precision potential.

The exact benchmark parameters are:

```text
mW = 80.4 GeV       mZ = 91.2 GeV       mh = 125.1 GeV
mt = 173.2 GeV      v = 246.2 GeV       Tc = 110.0 GeV
lambda_m = 1.5      lambda_s = 0.65
```

`benchmark.py` validates the model, optimizes an O(3) Fourier path, runs the
CosmoTransitions adapter, exports FindBounce geometry, and can invoke the
Wolfram driver. `findbounce.wls` defines the same potential in Wolfram and
runs genuine FindBounce from straight and Fourier-informed 31-point paths.
`__init__.py` makes the directory runnable as a Python module.

## Installation

From the repository root, install the base project for validation and the
Fourier-only stage:

```bash
pip install -e .
```

CosmoTransitions is optional and external:

```bash
pip install -e ".[cosmotransitions]"
```

The FindBounce calculation requires an external Mathematica or Wolfram Engine
installation with FindBounce installed. Neither final bounce solver is
bundled with this package.

## Commands

All commands below default to `T=85 GeV` and write generated files under the
selected output directory.

```bash
# Validate vacua, gradients, Hessians, boundedness, and Tc degeneracy.
python3 -m examples.xsm.benchmark validate --output-dir xsm_example_output

# Optimize and save the Fourier initializer and fixed-Vt diagnostic.
python3 -m examples.xsm.benchmark fourier --output-dir xsm_example_output

# Run genuine CosmoTransitions from straight and Fourier initializers.
python3 -m examples.xsm.benchmark cosmotransitions --output-dir xsm_example_output

# Export the open Fourier path without requiring Wolfram.
python3 -m examples.xsm.benchmark findbounce-export --output-dir xsm_example_output

# Export the path and run genuine FindBounce through Wolfram.
python3 -m examples.xsm.benchmark findbounce --output-dir xsm_example_output

# Run validation, Fourier, CosmoTransitions, and FindBounce stages.
python3 -m examples.xsm.benchmark all --output-dir xsm_example_output
```

Use `--temperature` for another temperature and `--help` for the complete
interface. After `findbounce-export`, the Wolfram stage can also be run from
the repository root with:

```bash
XSM_EXAMPLE_OUTPUT="$PWD/xsm_example_output" \
FOURIER_PATH_BOUNCE_WOLFRAM="$PWD/fourier_path_bounce/wolfram/FourierPathBounce.wl" \
wolframscript -file examples/xsm/findbounce.wls
```

The output directory contains model validation JSON, the serialized Fourier
result, full-precision path CSV, solver summaries, and solver paths/logs.
Around 85 GeV the final O(3) action is approximately `1.28e4`, or
`S3/T approximately 151`; small differences depend on solver discretization.

The Fourier `action_proxy` is an approximate fixed-`V_t` diagnostic used to
choose a curved initializer. It is not the final physical bounce action.
CosmoTransitions and FindBounce independently relax the supplied geometry and
compute the final action. Fourier mode count `N_m` is also distinct from the
`K=29` interior points exported to FindBounce.

References:

- P. Athron et al., *BubbleProfiler: finding the field profile and action for
  cosmological phase transitions*, arXiv:1901.03714, Section 8.
- V. Guada, M. Nemevsek and M. Pintar, *FindBounce: package for multi-field
  bounce actions*, arXiv:2002.00881, Section 5.6.
