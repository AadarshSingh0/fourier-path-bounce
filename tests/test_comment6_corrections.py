"""Regression tests for the Comment 6 audit corrections.

Two defects are covered:

* the Wolfram wrapper reported ``Status="ok"`` whenever FindBounce returned a
  finite action, so a run truncated at ``MaxPathIterations`` was indistinguishable
  from one that met the solver's own tolerance;
* ``ModeSelectionSettings.validate`` rejected a configuration whose only start
  point was the caller's ``initial_coefficients``;
* a D=3 optimization in which L-BFGS-B never left its start point was reported
  as ``optimizer_success=True`` and ``adaptive_converged=True``.
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.curved_valley_potential import (
    FALSE_VACUUM as VALLEY_FALSE,
    TRUE_VACUUM as VALLEY_TRUE,
    potential as valley_potential,
)
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

    def test_zero_column_coefficients_are_rejected(self):
        """(n_fields, 0) is not a usable coefficient array; reconstruct_path
        already rejects it, so optimize_fourier_path must agree."""
        settings = FourierPathSettings(
            mode_selection=ModeSelectionSettings(modes=(1,), **self.ONLY_USER)
        )
        with self.assertRaises(InputValidationError) as caught:
            optimize_fourier_path(
                potential, FALSE_VACUUM, TRUE_VACUUM,
                settings=settings, initial_coefficients=np.zeros((2, 0)),
            )
        self.assertIn("n_initial_modes >= 1", str(caught.exception))

    def test_default_validation_is_unchanged(self):
        FourierPathSettings().validate()
        ModeSelectionSettings().validate()


class NoProgressReportingTests(unittest.TestCase):
    """A run that never leaves its start point must not be called converged.

    Smallest reproduction of audit finding D3-1: the shipped curved-valley
    potential at D=3, n_grid=6, a single mode and library-default starts. The
    straight path is not stationary there (the projected gradient is O(100) while
    ``gtol`` is 1e-7), so the returned zero coefficients are an optimizer failure,
    not a solution.
    """

    SETTINGS = FourierPathSettings(
        dimension=3,
        n_grid=6,
        mode_selection=ModeSelectionSettings(modes=(1, 2, 3, 4, 5), patience=3),
    )

    def test_stalled_run_is_reported_as_failure_not_convergence(self):
        result = optimize_fourier_path(
            valley_potential, VALLEY_FALSE, VALLEY_TRUE, settings=self.SETTINGS
        )
        # Precondition: this configuration really does return the straight line.
        self.assertTrue(np.all(result.coefficients == 0.0))
        self.assertGreater(result.gradient_norm, self.SETTINGS.optimizer.gtol)
        # The three reported flags must all say so.
        self.assertTrue(result.optimizer_made_no_progress)
        self.assertFalse(result.optimizer_success)
        self.assertFalse(result.adaptive_converged)
        self.assertIn("no progress", result.stop_reason)
        self.assertIn("random_starts", result.stop_reason)
        self.assertIn("optimizer_made_no_progress", result.summary())

    def test_flag_survives_a_serialization_round_trip(self):
        from fourier_path_bounce import load_fourier_result, save_fourier_result

        result = optimize_fourier_path(
            valley_potential, VALLEY_FALSE, VALLEY_TRUE, settings=self.SETTINGS
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stalled.npz"
            save_fourier_result(result, path)
            restored = load_fourier_result(path)
        self.assertTrue(restored.optimizer_made_no_progress)
        self.assertFalse(restored.optimizer_success)
        self.assertTrue(np.array_equal(restored.path_points, result.path_points))

    def test_a_run_that_does_optimize_is_not_flagged(self):
        """No false positive: random restarts escape the stall on the same case."""
        settings = FourierPathSettings(
            dimension=3,
            n_grid=6,
            mode_selection=ModeSelectionSettings(
                modes=(1, 2, 3, 4, 5), patience=3, random_starts=3, random_scale=0.5
            ),
        )
        result = optimize_fourier_path(
            valley_potential, VALLEY_FALSE, VALLEY_TRUE, settings=settings
        )
        self.assertFalse(result.optimizer_made_no_progress)
        self.assertTrue(result.optimizer_success)
        self.assertGreater(result.coefficient_norm, 0.0)
        self.assertLess(result.action_proxy, 54.0)

    def test_healthy_d4_run_is_not_flagged(self):
        result = optimize_fourier_path(
            potential,
            FALSE_VACUUM,
            TRUE_VACUUM,
            settings=FourierPathSettings(
                dimension=4,
                n_grid=40,
                mode_selection=ModeSelectionSettings(modes=(1, 2), patience=2),
            ),
        )
        self.assertFalse(result.optimizer_made_no_progress)
        self.assertTrue(result.optimizer_success)


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
    def test_finite_action_without_diagnostics_is_not_reported_as_ok(self):
        """A finite action whose termination cannot be classified is 'unverified'.

        Exercised through the status logic directly: PathConvergence 'unknown'
        must downgrade 'ok' just as a reached limit does.
        """
        code = (
            f'Get["{WRAPPER.as_posix()}"];'
            'r=FourierPathBounce`FindBounceTerminationReport[<|"Action"->1.0|>,'
            '{"MaxPathIterations"->10},2];'
            'Print["CONV=",r["PathConvergence"]];'
            'Print["LIMIT=",r["PathIterationLimitReached"]];'
            'status="ok";'
            'If[status==="ok"&&TrueQ[r["PathIterationLimitReached"]],status="returned_at_path_iteration_limit"];'
            'If[status==="ok"&&r["PathConvergence"]==="unknown",status="termination_unverified"];'
            'Print["STATUS=",status];'
        )
        process = self._run(code, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("CONV=unknown", process.stdout)
        self.assertIn("LIMIT=False", process.stdout)
        self.assertIn("STATUS=termination_unverified", process.stdout)
        self.assertNotIn("STATUS=ok", process.stdout)

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
