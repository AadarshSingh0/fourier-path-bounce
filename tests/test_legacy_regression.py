import json
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fourier_path_bounce import FourierPathSettings, ModeSelectionSettings, reconstruct_path
from fourier_path_bounce.core import _build_action


ROOT = Path(__file__).resolve().parents[1]


class LegacyRegressionTests(unittest.TestCase):
    def test_path_formula_matches_legacy_exactly(self):
        false = np.array([0.0, -0.5, 0.25])
        true = np.array([1.0, 0.75, -0.25])
        coefficients = np.arange(9.0).reshape(3, 3) / 80.0
        t = np.linspace(0.0, 1.0, 260)
        basis = np.array([np.sin((k + 1) * np.pi * t) for k in range(3)])
        legacy = false[None, :] + t[:, None] * (true - false)[None, :] + basis.T @ coefficients.T
        legacy[0] = false
        legacy[-1] = true
        np.testing.assert_array_equal(
            reconstruct_path(false, true, coefficients, t), legacy
        )

    def test_saved_published_d4_action(self):
        fixture_file = ROOT / "tests/fixtures/d4_n4_legacy_regression.json"
        fixture = json.loads(fixture_file.read_text(encoding="ascii"))
        c = jnp.asarray(fixture["c"])
        delta = fixture["delta"]

        def potential(phi):
            return (jnp.sum(c * (phi - 1.0) ** 2) - delta) * jnp.sum(phi**2)

        flattened = np.asarray(fixture["coefficients"], dtype=float)
        modes = int(fixture["selected_modes"])
        coefficients = flattened.reshape(4, modes)
        settings = FourierPathSettings(
            dimension=4,
            n_grid=260,
            mode_selection=ModeSelectionSettings(modes=(modes,)),
        )
        objective, _ = _build_action(
            potential,
            np.asarray(fixture["false_vacuum"], dtype=float),
            np.asarray(fixture["true_vacuum"], dtype=float),
            fixture["potential_false"],
            fixture["potential_true"],
            settings,
            modes,
        )
        action, _ = objective(coefficients.ravel())
        expected = float(fixture["action"])
        self.assertLessEqual(abs(action - expected) / abs(expected), 1.0e-13)


if __name__ == "__main__":
    unittest.main()
