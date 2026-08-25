import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, potential
from fourier_path_bounce import (
    FourierPathSettings,
    InputValidationError,
    ModeResult,
    ModeSelectionSettings,
    OptimizerSettings,
    export_findbounce_points,
    load_findbounce_points,
    load_fourier_result,
    optimize_fourier_path,
    prepare_cosmotransitions_path,
    prepare_findbounce_points,
    save_fourier_result,
)


class SerializationAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = optimize_fourier_path(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            settings=FourierPathSettings(
                n_grid=40,
                optimizer=OptimizerSettings(maxiter=60),
                mode_selection=ModeSelectionSettings(modes=(1, 2), patience=2),
            ),
        )

    def test_result_serialization_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_fourier_result(self.result, Path(directory) / "result.npz")
            loaded = load_fourier_result(path)
        self.assertEqual(loaded.summary(), self.result.summary())
        np.testing.assert_array_equal(loaded.coefficients, self.result.coefficients)
        np.testing.assert_array_equal(loaded.path_points, self.result.path_points)

    def test_k_is_distinct_from_fourier_mode_count(self):
        one = prepare_findbounce_points(self.result, 1)
        many = prepare_findbounce_points(self.result, 4)
        self.assertEqual(one.K, 1)
        self.assertEqual(many.K, 4)
        self.assertEqual(one.selected_fourier_modes, self.result.selected_modes)
        self.assertEqual(many.selected_fourier_modes, self.result.selected_modes)
        self.assertEqual(one.interior_points.shape, (1, 2))
        self.assertEqual(many.interior_points.shape, (4, 2))

    def test_findbounce_export_and_round_trip(self):
        point_set = prepare_findbounce_points(self.result, 3, sampling="arclength")
        with tempfile.TemporaryDirectory() as directory:
            files = export_findbounce_points(point_set, directory)
            metadata = json.loads(files["metadata"].read_text())
            loaded = load_findbounce_points(files["metadata"])
        self.assertEqual(metadata["polygon"], "open")
        self.assertEqual(metadata["orientation"], "false_to_true")
        self.assertEqual(metadata["K"], 3)
        self.assertEqual(metadata["selected_fourier_modes"], self.result.selected_modes)
        np.testing.assert_array_equal(loaded.full_path, point_set.full_path)
        self.assertFalse(np.any(np.all(loaded.interior_points == FALSE_VACUUM, axis=1)))
        self.assertFalse(np.any(np.all(loaded.interior_points == TRUE_VACUUM, axis=1)))

    def test_invalid_k_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(InputValidationError):
                prepare_findbounce_points(self.result, value)

    def test_cosmotransitions_conversion_reverses_only_at_boundary(self):
        path = prepare_cosmotransitions_path(
            FALSE_VACUUM,
            TRUE_VACUUM,
            fourier_result=self.result,
            n_points=23,
        )
        self.assertEqual(path.shape, (23, 2))
        np.testing.assert_array_equal(path[0], TRUE_VACUUM)
        np.testing.assert_array_equal(path[-1], FALSE_VACUUM)
        np.testing.assert_array_equal(self.result.path_points[0], FALSE_VACUUM)
        np.testing.assert_array_equal(self.result.path_points[-1], TRUE_VACUUM)

    def test_cosmotransitions_endpoint_mismatch_rejected(self):
        with self.assertRaises(InputValidationError):
            prepare_cosmotransitions_path(
                FALSE_VACUUM,
                TRUE_VACUUM + 0.1,
                fourier_result=self.result,
                n_points=20,
            )


if __name__ == "__main__":
    unittest.main()
