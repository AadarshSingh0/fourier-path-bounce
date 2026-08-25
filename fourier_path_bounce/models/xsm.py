"""Leading high-temperature Z2-symmetric real-singlet Standard Model.

This is the two-field approximation used for the phenomenological examples in
BubbleProfiler (arXiv:1901.03714, Sec. 8) and FindBounce
(arXiv:2002.00881, Sec. 5.6). It is not a full one-loop precision potential.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class XSMParameters:
    """Published benchmark parameters.

    Masses, the electroweak scale, and temperatures are in GeV; quartic
    couplings are dimensionless.
    """
    mW: float = 80.4
    mZ: float = 91.2
    mh: float = 125.1
    mt: float = 173.2
    v: float = 246.2
    Tc: float = 110.0
    lambda_m: float = 1.5
    lambda_s: float = 0.65

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("xSM parameters must be finite")
        if min(self.mW, self.mZ, self.mh, self.mt, self.v, self.Tc) <= 0.0:
            raise ValueError("masses, v, and Tc must be positive")
        if self.lambda_s <= 0.0:
            raise ValueError("lambda_s must be positive")
        if self.lambda_m <= -2.0 * np.sqrt(self.lambda_h * self.lambda_s):
            raise ValueError("quartic potential is not bounded from below")

    @property
    def lambda_h(self) -> float:
        return self.mh**2 / (2.0 * self.v**2)

    @property
    def c_h(self) -> float:
        return (
            (2.0 * self.mW**2 + self.mZ**2 + self.mh**2 + 2.0 * self.mt**2)
            / (4.0 * self.v**2)
            + self.lambda_m / 24.0
        )

    @property
    def c_s(self) -> float:
        return (2.0 * self.lambda_m + 3.0 * self.lambda_s) / 12.0

    def to_dict(self) -> dict[str, float]:
        result = {key: float(value) for key, value in asdict(self).items()}
        result.update(lambda_h=self.lambda_h, c_h=self.c_h, c_s=self.c_s)
        return result


DEFAULT_XSM_PARAMETERS = XSMParameters()


class XSMThermalModel:
    """Authoritative NumPy/JAX-compatible implementation of the benchmark."""

    def __init__(self, parameters: XSMParameters = DEFAULT_XSM_PARAMETERS):
        parameters.validate()
        self.parameters = parameters

    def v_h_squared(self, temperature: float) -> float:
        """Return the squared electroweak minimum coordinate in GeV^2."""
        p = self.parameters
        value = p.v**2 * (1.0 - 2.0 * p.c_h * float(temperature) ** 2 / p.mh**2)
        if not np.isfinite(value):
            raise ValueError("v_h^2 is non-finite")
        return float(value)

    def v_s_squared(self, temperature: float) -> float:
        """Return the squared positive-singlet minimum coordinate in GeV^2."""
        p = self.parameters
        value = (
            (p.mh / (2.0 * p.v)) * self.v_h_squared(p.Tc) * np.sqrt(2.0 * p.lambda_s)
            + p.c_s * p.Tc**2
            - p.c_s * float(temperature) ** 2
        ) / p.lambda_s
        if not np.isfinite(value):
            raise ValueError("v_s^2 is non-finite")
        return float(value)

    def _thermal_masses(self, temperature: float) -> tuple[float, float]:
        p = self.parameters
        return (
            p.lambda_h * self.v_h_squared(temperature),
            -p.lambda_s * self.v_s_squared(temperature),
        )

    def vacua(self, temperature: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(false, true)`` in public false-to-true orientation."""
        vh2 = self.v_h_squared(temperature)
        vs2 = self.v_s_squared(temperature)
        if vh2 <= 0.0 or vs2 <= 0.0:
            raise ValueError("both analytic vacuum coordinates must be real and nonzero")
        false = np.array([0.0, np.sqrt(vs2)], dtype=float)
        true = np.array([np.sqrt(vh2), 0.0], dtype=float)
        return false, true

    def potential(self, point: Any, temperature: float) -> Any:
        """Evaluate V in GeV^4 for a field point in GeV.

        The field arithmetic remains differentiable when point is a JAX array.
        """
        h, s = point[0], point[1]
        p = self.parameters
        mu_h2, mu_s2 = self._thermal_masses(temperature)
        return (
            -0.5 * mu_h2 * h**2
            + 0.5 * mu_s2 * s**2
            + 0.25 * p.lambda_h * h**4
            + 0.25 * p.lambda_s * s**4
            + 0.25 * p.lambda_m * h**2 * s**2
        )

    def gradient(self, point: Any, temperature: float) -> np.ndarray:
        """Return the analytic field gradient in GeV^3."""
        h, s = np.asarray(point, dtype=float)
        p = self.parameters
        mu_h2, mu_s2 = self._thermal_masses(temperature)
        return np.array(
            [
                -mu_h2 * h + p.lambda_h * h**3 + 0.5 * p.lambda_m * h * s**2,
                mu_s2 * s + p.lambda_s * s**3 + 0.5 * p.lambda_m * h**2 * s,
            ],
            dtype=float,
        )

    def hessian(self, point: Any, temperature: float) -> np.ndarray:
        """Return the analytic field Hessian in GeV^2."""
        h, s = np.asarray(point, dtype=float)
        p = self.parameters
        mu_h2, mu_s2 = self._thermal_masses(temperature)
        return np.array(
            [
                [-mu_h2 + 3.0 * p.lambda_h * h**2 + 0.5 * p.lambda_m * s**2, p.lambda_m * h * s],
                [p.lambda_m * h * s, mu_s2 + 3.0 * p.lambda_s * s**2 + 0.5 * p.lambda_m * h**2],
            ],
            dtype=float,
        )

    def potential_callable(self, temperature: float) -> Callable[[Any], Any]:
        self.vacua(temperature)
        return lambda point: self.potential(point, temperature)

    def gradient_callable(self, temperature: float) -> Callable[[Any], np.ndarray]:
        self.vacua(temperature)
        return lambda point: self.gradient(point, temperature)

    def validate_temperature(
        self,
        temperature: float,
        tolerance: float = 1.0e-8,
    ) -> dict[str, Any]:
        """Validate stationarity and local stability at one temperature."""
        false, true = self.vacua(temperature)
        grad_false = self.gradient(false, temperature)
        grad_true = self.gradient(true, temperature)
        eig_false = np.linalg.eigvalsh(self.hessian(false, temperature))
        eig_true = np.linalg.eigvalsh(self.hessian(true, temperature))
        v_false = float(self.potential(false, temperature))
        v_true = float(self.potential(true, temperature))
        scale = max(1.0, abs(v_false), abs(v_true))
        stationary = (
            max(np.linalg.norm(grad_false), np.linalg.norm(grad_true))
            <= tolerance * scale
        )
        valid = stationary and np.min(eig_false) > 0.0 and np.min(eig_true) > 0.0
        return {
            "temperature_GeV": float(temperature),
            "false_vacuum": false.tolist(),
            "true_vacuum": true.tolist(),
            "potential_false_GeV4": v_false,
            "potential_true_GeV4": v_true,
            "delta_V_true_minus_false_GeV4": v_true - v_false,
            "gradient_false_norm": float(np.linalg.norm(grad_false)),
            "gradient_true_norm": float(np.linalg.norm(grad_true)),
            "hessian_false_eigenvalues_GeV2": eig_false.tolist(),
            "hessian_true_eigenvalues_GeV2": eig_true.tolist(),
            "valid_local_minima": bool(valid),
        }

    def wolfram_definitions(self, temperature: float) -> dict[str, Any]:
        """Return numeric inputs used to build the identical Wolfram expression."""
        false, true = self.vacua(temperature)
        mu_h2, mu_s2 = self._thermal_masses(temperature)
        return {
            **self.parameters.to_dict(),
            "temperature": float(temperature),
            "mu_h_squared": float(mu_h2),
            "mu_s_squared": float(mu_s2),
            "false_vacuum": false.tolist(),
            "true_vacuum": true.tolist(),
            "orientation": "false_to_true",
        }


__all__ = ["DEFAULT_XSM_PARAMETERS", "XSMParameters", "XSMThermalModel"]
