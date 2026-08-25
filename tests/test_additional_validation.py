import unittest

import jax.numpy as jnp
import numpy as np

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, potential
from fourier_path_bounce import (
    CosmoTransitionsSettings,
    FourierPathSettings,
    InputValidationError,
    ModeSelectionSettings,
    OptimizerSettings,
    PotentialEvaluationError,
    optimize_fourier_path,
    run_cosmotransitions,
)


class AdditionalValidationTests(unittest.TestCase):
    def test_three_field_optimization(self):
        false = np.append(FALSE_VACUUM, 0.0)
        true = np.append(TRUE_VACUUM, 0.0)

        def potential3(point):
            return potential(point[:2]) + 0.5 * point[2] ** 2

        result = optimize_fourier_path(
            potential3,
            false,
            true,
            settings=FourierPathSettings(
                n_grid=30,
                optimizer=OptimizerSettings(maxiter=40),
                mode_selection=ModeSelectionSettings(modes=(1,), patience=1),
            ),
        )
        self.assertEqual(result.n_fields, 3)
        self.assertEqual(result.coefficients.shape, (3, 1))
        np.testing.assert_array_equal(result.path_points[0], false)
        np.testing.assert_array_equal(result.path_points[-1], true)

    def test_numpy_only_potential_gets_clear_jax_error(self):
        def numpy_only(point):
            values = np.asarray(point, dtype=float)
            return np.sum((values - 1.0) ** 2)

        with self.assertRaisesRegex(PotentialEvaluationError, "differentiable by JAX"):
            optimize_fourier_path(
                numpy_only,
                [0.0, 0.0],
                [1.0, 1.0],
                settings=FourierPathSettings(
                    n_grid=20,
                    optimizer=OptimizerSettings(maxiter=5),
                    mode_selection=ModeSelectionSettings(modes=(1,), patience=1),
                ),
            )

    def test_malformed_gradient_rejected(self):
        with self.assertRaises(PotentialEvaluationError):
            optimize_fourier_path(
                potential,
                FALSE_VACUUM,
                TRUE_VACUUM,
                potential_gradient=lambda _point: np.array([1.0]),
            )

    def test_cosmotransitions_dimension_mismatch_rejected_before_solver(self):
        result = optimize_fourier_path(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            settings=FourierPathSettings(
                dimension=4,
                n_grid=24,
                optimizer=OptimizerSettings(maxiter=20),
                mode_selection=ModeSelectionSettings(modes=(1,), patience=1),
            ),
        )
        with self.assertRaisesRegex(InputValidationError, "dimensions must match"):
            run_cosmotransitions(
                potential,
                lambda point: np.zeros_like(point),
                FALSE_VACUUM,
                TRUE_VACUUM,
                fourier_result=result,
                settings=CosmoTransitionsSettings(dimension=3),
            )


if __name__ == "__main__":
    unittest.main()
