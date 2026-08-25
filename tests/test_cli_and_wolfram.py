import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, potential
from fourier_path_bounce import (
    FourierPathSettings,
    ModeSelectionSettings,
    OptimizerSettings,
    export_findbounce_points,
    optimize_fourier_path,
    prepare_findbounce_points,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class CliTests(unittest.TestCase):
    def test_help(self):
        process = subprocess.run(
            [PYTHON, "-m", "fourier_path_bounce", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("findbounce-export", process.stdout)
        self.assertIn("ct-prepare", process.stdout)

    def test_small_successful_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            optimize = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "fourier_path_bounce",
                    "optimize",
                    "--potential",
                    "examples.reusable_potential:potential",
                    "--false",
                    ",".join(map(str, FALSE_VACUUM)),
                    "--true",
                    ",".join(map(str, TRUE_VACUUM)),
                    "--n-grid",
                    "32",
                    "--modes",
                    "1",
                    "--maxiter",
                    "30",
                    "--patience",
                    "1",
                    "--output-dir",
                    str(destination / "optimized"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(optimize.returncode, 0, optimize.stderr)
            result_file = destination / "optimized" / "fourier_result.npz"
            self.assertTrue(result_file.exists())
            export = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "fourier_path_bounce",
                    "findbounce-export",
                    "--result",
                    str(result_file),
                    "--K",
                    "2",
                    "--sampling",
                    "arclength",
                    "--output-dir",
                    str(destination / "findbounce"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(export.returncode, 0, export.stderr)
            metadata = json.loads(
                (destination / "findbounce" / "findbounce_fourier_metadata.json").read_text()
            )
            self.assertEqual(metadata["K"], 2)


class WolframImportTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("wolframscript"), "wolframscript is unavailable")
    def test_generic_wolfram_import(self):
        result = optimize_fourier_path(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            settings=FourierPathSettings(
                n_grid=30,
                optimizer=OptimizerSettings(maxiter=30),
                mode_selection=ModeSelectionSettings(modes=(1,), patience=1),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            files = export_findbounce_points(
                prepare_findbounce_points(result, 2), directory, basename="wltest"
            )
            code = (
                f'Get["{(ROOT / "fourier_path_bounce/wolfram/FourierPathBounce.wl").as_posix()}"];'
                f'p=FourierPathBounce`ImportFourierPathPoints["{files["metadata"].as_posix()}",'
                f'{_wl_list(FALSE_VACUUM)},{_wl_list(TRUE_VACUUM)}];'
                'If[MatrixQ[p,NumericQ]&&Dimensions[p]=={4,2},Print["WOLFRAM_IMPORT_OK"],Exit[4]];'
            )
            process = subprocess.run(
                ["wolframscript", "-code", code],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=60,
            )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("WOLFRAM_IMPORT_OK", process.stdout)


def _wl_list(values):
    return "{" + ",".join(f"{float(value):.17g}" for value in values) + "}"


if __name__ == "__main__":
    unittest.main()
