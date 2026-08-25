# D=4 regression fixture

`d4_n4_legacy_regression.json` stores the full-precision coefficients,
potential parameters, vacua, selected mode count, and published D=4 action used
by `tests/test_legacy_regression.py`.

Its numerical precision is part of the regression contract and must not be
rounded. Update this fixture only after a deliberate scientific change has
been independently validated.
