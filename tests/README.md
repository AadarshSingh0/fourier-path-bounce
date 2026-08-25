# Reusable package tests

This folder contains focused unit, integration-boundary, and regression tests
for the reusable Fourier preconditioning package.

## File map

- `__init__.py` makes the test suite importable for targeted
  `unittest` commands.
- `test_core.py` checks endpoint-preserving path geometry, validation,
  deterministic optimization, sampling, and the validated three- and
  four-dimensional action arithmetic.
- `test_serialization_and_adapters.py` checks result round trips,
  FindBounce exports, the distinction between `N_m` and `K`, and
  CosmoTransitions boundary orientation.
- `test_cli_and_wolfram.py` checks CLI help and a small end-to-end export,
  then validates the generic Wolfram path importer when `wolframscript`
  is available. The Wolfram-specific test is skipped automatically when it is
  unavailable.
- `test_legacy_regression.py` checks exact legacy path construction and
  evaluates the stored full-precision published D=4 action fixture.
- `test_additional_validation.py` covers arbitrary field dimension,
  malformed gradients, incompatible solver dimensions, and clear rejection of
  non-JAX potential functions.
- `test_xsm_model.py` validates the finite-temperature real-singlet model,
  including its analytic minima and derivatives, NumPy/JAX/Wolfram agreement,
  and the portable example command interface.
- `fixtures/d4_n4_legacy_regression.json` stores coefficients, potential
  data, endpoints, and the full-precision expected D=4 action used by the
  legacy regression test.

## Running the tests

Install the project first. Python 3.10+, JAX, NumPy, and SciPy are required;
the optional Wolfram check additionally needs `wolframscript`.

Focused reusable-package suite:

```bash
python3 -m unittest discover -s tests -v
```

Repository-wide discovery:

```bash
python3 -m unittest discover -v
```

The regression fixture preserves a published result at full precision. Do not
casually regenerate, round, or hand-edit it; update it only after a deliberate,
independently validated scientific change.
