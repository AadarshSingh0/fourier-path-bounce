# Phenomenological models

`xsm.py` implements the finite-temperature, Z2-symmetric real-singlet
extension used in BubbleProfiler (arXiv:1901.03714, Section 8) and FindBounce
(arXiv:2002.00881, Section 5.6). It provides one authoritative parameter set,
the potential, analytic gradient and Hessian, analytic vacua, validation
diagnostics, and numeric data for the Wolfram boundary.

```python
from fourier_path_bounce.models import XSMThermalModel

model = XSMThermalModel()
false_vacuum, true_vacuum = model.vacua(85.0)
potential = model.potential_callable(85.0)
gradient = model.gradient_callable(85.0)
```

The model is the leading high-temperature approximation used in those
references. It is not a full one-loop or daisy-resummed precision potential.
