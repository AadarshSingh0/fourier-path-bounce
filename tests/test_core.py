import math
import unittest

import jax.numpy as jnp
import numpy as np

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, gradient, potential
from fourier_path_bounce import (
    FourierPathSettings,
    InputValidationError,
    ModeSelectionSettings,
    OptimizerSettings,
    PotentialEvaluationError,
    optimize_fourier_path,
    reconstruct_path,
)
from fourier_path_bounce.core import _build_action, action_prefactor


class CoreGeometryTests(unittest.TestCase):
    def test_endpoint_preservation_and_arbitrary_field_dimension(self):
        false = np.zeros(4)
        true = np.arange(1.0, 5.0)
        coefficients = np.arange(12.0).reshape(4, 3) / 100.0
        path = reconstruct_path(false, true, coefficients, np.linspace(0.0, 1.0, 31))
        self.assertEqual(path.shape, (31, 4))
        np.testing.assert_array_equal(path[0], false)
        np.testing.assert_array_equal(path[-1], true)

    def test_endpoint_dimension_mismatch(self):
        with self.assertRaises(InputValidationError):
            reconstruct_path([0.0, 0.0], [1.0], np.zeros((2, 1)), [0.0, 1.0])

    def test_nonfinite_endpoint_rejected(self):
        with self.assertRaises(InputValidationError):
            reconstruct_path([0.0, np.nan], [1.0, 1.0], np.zeros((2, 1)), [0.0, 1.0])

    def test_invalid_potential_output_rejected(self):
        def bad(_point):
            return jnp.array([1.0, 2.0])

        settings = FourierPathSettings(
            n_grid=20,
            mode_selection=ModeSelectionSettings(modes=(1,), patience=1),
            optimizer=OptimizerSettings(maxiter=5),
        )
        with self.assertRaises(PotentialEvaluationError):
            optimize_fourier_path(bad, [0.0, 0.0], [1.0, 1.0], settings=settings)

    def test_nonfinite_potential_rejected(self):
        def bad(_point):
            return jnp.nan

        with self.assertRaises(PotentialEvaluationError):
            optimize_fourier_path(bad, [0.0], [1.0])


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = FourierPathSettings(
            n_grid=48,
            optimizer=OptimizerSettings(maxiter=80),
            mode_selection=ModeSelectionSettings(modes=(1,), patience=1, seed=17),
        )
        cls.first = optimize_fourier_path(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            settings=cls.settings,
            potential_gradient=gradient,
        )

    def test_shapes_and_orientation(self):
        result = self.first
        self.assertEqual(result.coefficients.shape, (2, 1))
        self.assertEqual(result.path_points.shape, (48, 2))
        self.assertEqual(result.orientation, "false_to_true")
        np.testing.assert_array_equal(result.path_points[0], FALSE_VACUUM)
        np.testing.assert_array_equal(result.path_points[-1], TRUE_VACUUM)
        self.assertTrue(math.isfinite(result.action_proxy))

    def test_deterministic_reproducibility(self):
        second = optimize_fourier_path(
            potential, FALSE_VACUUM, TRUE_VACUUM, settings=self.settings
        )
        self.assertEqual(self.first.action_proxy, second.action_proxy)
        np.testing.assert_array_equal(self.first.coefficients, second.coefficients)
        np.testing.assert_array_equal(self.first.path_points, second.path_points)

    def test_parameter_and_arclength_sampling(self):
        parameter = self.first.sample(17, sampling="parameter")
        arclength = self.first.sample(17, sampling="arclength")
        self.assertEqual(parameter.shape, (17, 2))
        self.assertEqual(arclength.shape, (17, 2))
        np.testing.assert_array_equal(parameter[[0, -1]], arclength[[0, -1]])
        arc_segments = np.linalg.norm(np.diff(arclength, axis=0), axis=1)
        self.assertLess(np.std(arc_segments) / np.mean(arc_segments), 0.02)
        custom = self.first.sample_at([0.0, 0.1, 0.8, 1.0])
        self.assertEqual(custom.shape, (4, 2))
        np.testing.assert_array_equal(custom[0], FALSE_VACUUM)
        np.testing.assert_array_equal(custom[-1], TRUE_VACUUM)

    def test_d4_action_matches_legacy_formula_without_clipping(self):
        coefficients = np.array([[0.04], [-0.03]])
        settings = FourierPathSettings(
            dimension=4,
            n_grid=64,
            mode_selection=ModeSelectionSettings(modes=(1,)),
        )
        objective, _ = _build_action(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            float(potential(FALSE_VACUUM)),
            float(potential(TRUE_VACUUM)),
            settings,
            1,
        )
        actual, _ = objective(coefficients.ravel())
        t = np.linspace(0.0, 1.0, 64)
        path = reconstruct_path(FALSE_VACUUM, TRUE_VACUUM, coefficients, t)
        v_false = float(potential(FALSE_VACUUM))
        v_true = float(potential(TRUE_VACUUM))
        values = np.array([float(potential(point)) for point in path]) - v_false
        vt = (v_true - v_false) * t * t * (3.0 - 2.0 * t)
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        vdiff = values[:-1] + values[1:] - vt[:-1] - vt[1:]
        expected = 0.5 * 27.0 * np.pi**2 * np.sum(
            vdiff**2 * ds**4 / (-(np.diff(vt) ** 3))
        )
        self.assertLessEqual(abs(actual - expected) / abs(expected), 1.0e-14)

    def test_d3_action_matches_inherited_regulator(self):
        coefficients = np.array([[0.02], [-0.01]])
        settings = FourierPathSettings(
            dimension=3,
            n_grid=64,
            mode_selection=ModeSelectionSettings(modes=(1,)),
        )
        objective, _ = _build_action(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            float(potential(FALSE_VACUUM)),
            float(potential(TRUE_VACUUM)),
            settings,
            1,
        )
        actual, _ = objective(coefficients.ravel())
        t = np.linspace(0.0, 1.0, 64)
        path = reconstruct_path(FALSE_VACUUM, TRUE_VACUUM, coefficients, t)
        v_false = float(potential(FALSE_VACUUM))
        v_true = float(potential(TRUE_VACUUM))
        values = np.array([float(potential(point)) for point in path]) - v_false
        vt = (v_true - v_false) * t * t * (3.0 - 2.0 * t)
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        vdiff = values[:-1] + values[1:] - vt[:-1] - vt[1:]
        minus_dvt = -np.diff(vt)
        terms = np.maximum(vdiff, 1e-300) ** 1.5 * ds**3 / np.maximum(minus_dvt, 1e-300) ** 2
        expected = action_prefactor(3) * np.sum(terms)
        if np.any((vdiff <= 0.0) | (minus_dvt <= 0.0) | ~np.isfinite(terms)):
            expected += 1e50
        self.assertLessEqual(abs(actual - expected) / abs(expected), 1.0e-14)


if __name__ == "__main__":
    unittest.main()
