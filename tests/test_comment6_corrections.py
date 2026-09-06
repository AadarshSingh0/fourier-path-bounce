"""Regression tests for the Comment 6 audit corrections.

Two defects are covered:

* the Wolfram wrapper reported ``Status="ok"`` whenever FindBounce returned a
  finite action, so a run truncated at ``MaxPathIterations`` was indistinguishable
  from one that met the solver's own tolerance;
* ``ModeSelectionSettings.validate`` rejected a configuration whose only start
  point was the caller's ``initial_coefficients``.
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, potential
from fourier_path_bounce import (
    FourierPathSettings,
    InputValidationError,
    ModeSelectionSettings,
    OptimizerSettings,
    export_findbounce_points,
    optimize_fourier_path,
    prepare_findbounce_points,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "fourier_path_bounce" / "wolfram" / "FourierPathBounce.wl"

# Audited values for the documented K=4 example (FindBounce 1.1.0, Wolfram 13.2).
DOCUMENTED_K4_CAP = 10
DOCUMENTED_K4_ITERATIONS_AT_CAP = 10
SUFFICIENT_K4_CAP = 20
DOCUMENTED_K4_ITERATIONS_BELOW_CAP = 11


def _wl_list(values):
    return "{" + ",".join(f"{float(value):.17g}" for value in values) + "}"


class InitialCoefficientStartTests(unittest.TestCase):
    """`initial_coefficients` is a complete start strategy on its own."""

    ONLY_USER = dict(zero_start=False, warm_previous=False, warm_best=False, random_starts=0)

    def test_all_builtin_starts_disabled_still_rejected_without_user_start(self):
        settings = ModeSelectionSettings(**self.ONLY_USER)
        with self.assertRaises(InputValidationError) as caught:
            settings.validate()
        # The message must name the way out, including the user-start option.
        self.assertIn("initial_coefficients", str(caught.exception))

    def test_validate_accepts_user_start_as_sole_strategy(self):
        ModeSelectionSettings(**self.ONLY_USER).validate(external_start_available=True)
        FourierPathSettings(
            mode_selection=ModeSelectionSettings(**self.ONLY_USER)
        ).validate(external_start_available=True)

    def test_optimize_runs_with_user_coefficients_as_only_start(self):
        settings = FourierPathSettings(
            dimension=4,
            n_grid=40,
            optimizer=OptimizerSettings(maxiter=40),
            mode_selection=ModeSelectionSettings(modes=(1, 2), patience=2, **self.ONLY_USER),
        )
        result = optimize_fourier_path(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            settings=settings,
            initial_coefficients=np.array([[0.05], [0.10]]),
        )
        self.assertTrue(np.all(np.isfinite(result.coefficients)))
        for mode_result in result.history:
            self.assertEqual(set(mode_result.start_actions), {"user"})
            self.assertEqual(mode_result.start_label, "user")

    def test_same_settings_without_user_coefficients_raise_at_call_time(self):
        settings = FourierPathSettings(
            mode_selection=ModeSelectionSettings(modes=(1,), **self.ONLY_USER)
        )
        with self.assertRaises(InputValidationError):
            optimize_fourier_path(potential, FALSE_VACUUM, TRUE_VACUUM, settings=settings)

    def test_default_validation_is_unchanged(self):
        FourierPathSettings().validate()
        ModeSelectionSettings().validate()


class FindBounceTerminationReportingTests(unittest.TestCase):
    """A finite action alone must not be reported as convergence."""

    @staticmethod
    def _run(code, timeout=180):
        return subprocess.run(
            ["wolframscript", "-code", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    @unittest.skipUnless(shutil.which("wolframscript"), "wolframscript is unavailable")
    def test_unavailable_iteration_count_is_reported_as_unknown(self):
        """No FindBounce run required: $Failed must not be classified as converged."""
        code = (
            f'Get["{WRAPPER.as_posix()}"];'
            'r=FourierPathBounce`FindBounceTerminationReport[$Failed,'
            '{"MaxPathIterations"->10},2];'
            'Print[r["PathConvergence"]];'
            'Print[r["PathIterationLimitReached"]];'
        )
        process = self._run(code, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("unknown", process.stdout)
        self.assertIn("False", process.stdout)

    @unittest.skipUnless(shutil.which("wolframscript"), "wolframscript is unavailable")
    def test_single_field_limit_concept_does_not_apply(self):
        code = (
            f'Get["{WRAPPER.as_posix()}"];'
            'r=FourierPathBounce`FindBounceTerminationReport[$Failed,'
            '{"MaxPathIterations"->0},1];'
            'Print[r["PathConvergence"]];'
        )
        process = self._run(code, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("not_applicable_single_field", process.stdout)

    @unittest.skipUnless(shutil.which("wolframscript"), "wolframscript is unavailable")
    def test_documented_k4_example_reports_limit_and_convergence_distinctly(self):
        """cap 10 truncates and must say so; cap 20 finishes below the limit."""
        result = optimize_fourier_path(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            settings=FourierPathSettings(
                n_grid=130,
                mode_selection=ModeSelectionSettings(modes=(1, 2), patience=2),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            files = export_findbounce_points(
                prepare_findbounce_points(result, 4, sampling="arclength"),
                directory,
                basename="k4",
            )
            record = Path(directory) / "summary.json"
            template = (
                f'Get["{WRAPPER.as_posix()}"];'
                'pot=1/4 (x^2-1)^2+1/2 (y-0.35 (x^2-1))^2-0.1 x;'
                'r=FourierPathBounce`RunFindBounceWithFourier[pot,{x,y},'
                f'{_wl_list(FALSE_VACUUM)},{_wl_list(TRUE_VACUUM)},4,'
                f'"{files["metadata"].as_posix()}",'
                '"Gradient"->{D[pot,x],D[pot,y]},'
                '"FindBounceOptions"->{"MaxPathIterations"->%d,'
                '"MaxRadiusIterations"->200,"PathTolerance"->0.01},'
                '"TimeLimit"->120%s];'
                'Print["STATUS=",r["Status"]];'
                'Print["ITERS=",r["PathIterations"]];'
                'Print["CAP=",r["ConfiguredMaxPathIterations"]];'
                'Print["LIMIT=",r["PathIterationLimitReached"]];'
                'Print["ACTION=",InputForm[r["Action"]]];'
                'Print["HASBOUNCE=",r["Bounce"]=!=$Failed];'
            )
            truncated = self._run(
                template % (DOCUMENTED_K4_CAP, f',"ResultFile"->"{record.as_posix()}"')
            )
            self.assertEqual(truncated.returncode, 0, truncated.stdout + truncated.stderr)
            self.assertIn("STATUS=returned_at_path_iteration_limit", truncated.stdout)
            self.assertIn(f"ITERS={DOCUMENTED_K4_ITERATIONS_AT_CAP}", truncated.stdout)
            self.assertIn(f"CAP={DOCUMENTED_K4_CAP}", truncated.stdout)
            self.assertIn("LIMIT=True", truncated.stdout)
            # The action and the bounce object must survive a limit-reached run.
            self.assertIn("HASBOUNCE=True", truncated.stdout)

            # The result file must be a usable record, not a zero-byte stub.
            self.assertTrue(record.exists())
            self.assertGreater(record.stat().st_size, 0)
            payload = json.loads(record.read_text())
            self.assertEqual(payload["Status"], "returned_at_path_iteration_limit")
            self.assertEqual(payload["PathIterations"], DOCUMENTED_K4_ITERATIONS_AT_CAP)
            self.assertTrue(payload["PathIterationLimitReached"])

            sufficient = self._run(template % (SUFFICIENT_K4_CAP, ""))
            self.assertEqual(sufficient.returncode, 0, sufficient.stdout + sufficient.stderr)
            self.assertIn("STATUS=ok", sufficient.stdout)
            self.assertIn(f"ITERS={DOCUMENTED_K4_ITERATIONS_BELOW_CAP}", sufficient.stdout)
            self.assertIn("LIMIT=False", sufficient.stdout)


if __name__ == "__main__":
    unittest.main()
