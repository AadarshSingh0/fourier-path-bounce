import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root

from examples.xsm.benchmark import build_parser, validate_model
from fourier_path_bounce.models.xsm import DEFAULT_XSM_PARAMETERS, XSMThermalModel


class XSMModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = XSMThermalModel()
        cls.temperatures = (75.0, 80.0, 81.0, 85.0, 90.0, 95.0, 100.0)

    def test_quoted_thermal_coefficients(self):
        p = DEFAULT_XSM_PARAMETERS
        self.assertAlmostEqual(p.lambda_h, 0.12909473386209622, places=15)
        self.assertAlmostEqual(p.c_h, 0.46212527905891737, places=15)
        self.assertAlmostEqual(p.c_s, 0.4125, places=15)

    def test_quoted_vacua(self):
        false81, true81 = self.model.vacua(81.0)
        false85, true85 = self.model.vacua(85.0)
        np.testing.assert_allclose(true81, [192.685706, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(false81, [0.0, 105.947029], atol=1.0e-6)
        np.testing.assert_allclose(true85, [186.415760, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(false85, [0.0, 103.939350], atol=1.0e-6)

    def test_analytic_vacua_are_numerical_stationary_points(self):
        for temperature in self.temperatures:
            for vacuum in self.model.vacua(temperature):
                self.assertLess(np.linalg.norm(self.model.gradient(vacuum, temperature)), 1.0e-8)
                solution = root(lambda x: self.model.gradient(x, temperature), vacuum + [0.01, -0.01])
                self.assertTrue(solution.success)
                np.testing.assert_allclose(solution.x, vacuum, rtol=0.0, atol=1.0e-8)

    def test_vacua_are_minima_and_true_is_deeper_below_tc(self):
        for temperature in self.temperatures:
            validation = self.model.validate_temperature(temperature)
            self.assertTrue(validation["valid_local_minima"])
            self.assertGreater(min(validation["hessian_false_eigenvalues_GeV2"]), 0.0)
            self.assertGreater(min(validation["hessian_true_eigenvalues_GeV2"]), 0.0)
            self.assertLess(validation["delta_V_true_minus_false_GeV4"], 0.0)

    def test_degenerate_at_critical_temperature_and_bounded(self):
        p = self.model.parameters
        false, true = self.model.vacua(p.Tc)
        vf = float(self.model.potential(false, p.Tc))
        vt = float(self.model.potential(true, p.Tc))
        self.assertLessEqual(abs(vt - vf), 1.0e-12 * max(abs(vf), 1.0))
        self.assertGreater(p.lambda_h, 0.0)
        self.assertGreater(p.lambda_s, 0.0)
        self.assertGreater(p.lambda_m, -2.0 * np.sqrt(p.lambda_h * p.lambda_s))

    def test_numpy_jax_and_analytic_gradient_agree(self):
        for temperature in (81.0, 85.0, 90.0):
            for point in ([20.0, 30.0], [95.5, 44.25], [160.0, 5.0]):
                np_value = float(self.model.potential(np.asarray(point), temperature))
                jax_value = float(self.model.potential(jnp.asarray(point), temperature))
                self.assertAlmostEqual(np_value, jax_value, places=8)
                automatic = np.asarray(jax.grad(lambda x: self.model.potential(x, temperature))(jnp.asarray(point)))
                analytic = self.model.gradient(point, temperature)
                np.testing.assert_allclose(analytic, automatic, rtol=2.0e-13, atol=1.0e-9)
                eps = 1.0e-4
                finite = np.array([
                    (self.model.potential(np.asarray(point) + np.eye(2)[i] * eps, temperature)
                     - self.model.potential(np.asarray(point) - np.eye(2)[i] * eps, temperature)) / (2.0 * eps)
                    for i in range(2)
                ])
                np.testing.assert_allclose(analytic, finite, rtol=2.0e-7, atol=2.0e-4)

    def test_wolfram_export_is_complete_and_matches_model(self):
        data = self.model.wolfram_definitions(85.0)
        self.assertEqual(data["orientation"], "false_to_true")
        point = np.array([61.0, 72.0])
        p = self.model.parameters
        h, s = point
        value = (-0.5 * data["mu_h_squared"] * h**2 + 0.5 * data["mu_s_squared"] * s**2
                 + 0.25 * p.lambda_h * h**4 + 0.25 * p.lambda_s * s**4
                 + 0.25 * p.lambda_m * h**2 * s**2)
        self.assertAlmostEqual(value, float(self.model.potential(point, 85.0)), places=8)

    def test_public_example_parser_and_validation_stage(self):
        args = build_parser().parse_args(["validate"])
        self.assertEqual(args.temperature, 85.0)
        self.assertEqual(args.output_dir, Path("xsm_example_output"))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = validate_model(85.0, output_dir)
            self.assertTrue(result["valid_local_minima"])
            self.assertTrue(result["critical_vacua_degenerate"])
            self.assertTrue((output_dir / "validation.json").is_file())

    @unittest.skipUnless(shutil.which("wolframscript"), "wolframscript is unavailable")
    def test_genuine_wolfram_kernel_matches_python(self):
        data = self.model.wolfram_definitions(85.0)
        h, s = 61.0, 72.0
        expression = (
            f"-{data['mu_h_squared']:.17g} h^2/2+{data['mu_s_squared']:.17g} s^2/2+"
            f"{data['lambda_h']:.17g} h^4/4+{data['lambda_s']:.17g} s^4/4+"
            f"{data['lambda_m']:.17g} h^2 s^2/4"
        )
        code = (
            f"v={expression}; p={{h->{h:.17g},s->{s:.17g}}};"
            'Print[ExportString[<|"V"->N[v/.p],"G"->N[{D[v,h],D[v,s]}/.p]|>,"RawJSON"]]'
        )
        process = subprocess.run(
            ["wolframscript", "-code", code], text=True, capture_output=True,
            timeout=30, check=True,
        )
        payload = json.loads(process.stdout.rsplit("\nNull", 1)[0])
        np.testing.assert_allclose(payload["G"], self.model.gradient([h, s], 85.0), rtol=0.0, atol=1.0e-8)
        self.assertAlmostEqual(payload["V"], float(self.model.potential([h, s], 85.0)), places=7)


if __name__ == "__main__":
    unittest.main()
