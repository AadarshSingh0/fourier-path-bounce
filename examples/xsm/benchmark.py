"""Run the reusable finite-temperature xSM benchmark.

The Fourier calculation constructs an initializer and an approximate fixed-Vt
diagnostic. CosmoTransitions or FindBounce remains responsible for the final
physical O(3) bounce action.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np

from fourier_path_bounce import (
    CosmoTransitionsSettings,
    FourierPathSettings,
    ModeSelectionSettings,
    OptimizerSettings,
    export_findbounce_points,
    optimize_fourier_path,
    prepare_findbounce_points,
    run_cosmotransitions,
    save_fourier_result,
)
from fourier_path_bounce.models import XSMThermalModel


DEFAULT_TEMPERATURE_GEV = 85.0
DEFAULT_FIND_BOUNCE_INTERIOR_POINTS = 29


def fourier_settings() -> FourierPathSettings:
    """Return the deterministic O(3) settings used by this example."""
    return FourierPathSettings(
        dimension=3,
        n_grid=260,
        profile="cubic_smoothstep_raw_t",
        optimizer=OptimizerSettings(
            maxiter=800,
            ftol=1.0e-12,
            gtol=1.0e-7,
            maxls=100,
            coefficient_bound_factor=0.5,
        ),
        mode_selection=ModeSelectionSettings(
            modes=(1, 2, 3, 4, 5),
            relative_tolerance=1.0e-3,
            patience=3,
            zero_start=True,
            warm_previous=True,
            random_starts=3,
            random_scale=0.03,
            seed=20260825,
        ),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="ascii",
    )


def validate_model(temperature: float, output_directory: Path) -> dict[str, Any]:
    """Validate the two minima and write one portable JSON summary."""
    model = XSMThermalModel()
    validation = model.validate_temperature(temperature)
    critical = model.validate_temperature(model.parameters.Tc)
    tolerance = 1.0e-10 * max(
        1.0,
        abs(critical["potential_false_GeV4"]),
        abs(critical["potential_true_GeV4"]),
    )
    validation.update(
        parameters=model.parameters.to_dict(),
        quartic_bounded_from_below=True,
        critical_temperature_GeV=model.parameters.Tc,
        critical_vacuum_energy_difference_GeV4=critical[
            "delta_V_true_minus_false_GeV4"
        ],
        critical_vacua_degenerate=abs(
            critical["delta_V_true_minus_false_GeV4"]
        )
        <= tolerance,
        approximation="leading high-temperature",
    )
    if (
        not validation["valid_local_minima"]
        or validation["delta_V_true_minus_false_GeV4"] >= 0.0
        or not validation["critical_vacua_degenerate"]
    ):
        raise RuntimeError(
            f"the xSM endpoints are not valid tunnelling minima at T={temperature:g} GeV"
        )
    _write_json(output_directory / "validation.json", validation)
    return validation


def optimize_path(temperature: float, output_directory: Path):
    """Optimize and save the endpoint-preserving Fourier initializer."""
    model = XSMThermalModel()
    false_vacuum, true_vacuum = model.vacua(temperature)
    result = optimize_fourier_path(
        model.potential_callable(temperature),
        false_vacuum,
        true_vacuum,
        settings=fourier_settings(),
        potential_gradient=model.gradient_callable(temperature),
        metadata={
            "model": "finite_temperature_z2_xsm",
            "temperature_GeV": temperature,
        },
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    save_fourier_result(result, output_directory / "fourier_result.npz")
    np.savetxt(
        output_directory / "fourier_path.csv",
        np.column_stack((result.parameter, result.path_points)),
        delimiter=",",
        header="t,h_GeV,s_GeV",
        comments="",
        fmt="%.17g",
    )
    summary = result.summary()
    summary["mode_history"] = [asdict(item) for item in result.history]
    _write_json(output_directory / "fourier_summary.json", summary)
    return result


def run_ct(temperature: float, output_directory: Path) -> dict[str, Any]:
    """Run genuine CosmoTransitions from straight and Fourier initializers."""
    model = XSMThermalModel()
    false_vacuum, true_vacuum = model.vacua(temperature)
    fourier = optimize_path(temperature, output_directory)
    settings = CosmoTransitionsSettings(
        dimension=3,
        maxiter=40,
        fix_end_cutoff=0.03,
        v_spline_samples=100,
    )
    payload: dict[str, Any] = {
        "temperature_GeV": temperature,
        "solver": "CosmoTransitions",
        "version": distribution_version("cosmoTransitions"),
        "runs": {},
    }
    for initialization in ("straight", "fourier"):
        result = run_cosmotransitions(
            model.potential_callable(temperature),
            model.gradient_callable(temperature),
            false_vacuum,
            true_vacuum,
            fourier_result=fourier if initialization == "fourier" else None,
            initialization=initialization,
            n_path_points=120,
            sampling="arclength" if initialization == "fourier" else "parameter",
            settings=settings,
        )
        np.savetxt(
            output_directory / f"cosmotransitions_{initialization}_path.csv",
            result.final_path,
            delimiter=",",
            header="field_0_GeV,field_1_GeV",
            comments="",
            fmt="%.17g",
        )
        (output_directory / f"cosmotransitions_{initialization}.log").write_text(
            result.solver_log, encoding="utf-8"
        )
        payload["runs"][initialization] = result.summary()
        payload["runs"][initialization]["S3_over_T"] = result.action / temperature
    straight = payload["runs"]["straight"]["action"]
    fourier_action = payload["runs"]["fourier"]["action"]
    payload["relative_action_difference"] = (fourier_action - straight) / straight
    _write_json(output_directory / "cosmotransitions_results.json", payload)
    return payload


def export_findbounce(temperature: float, output_directory: Path) -> Path:
    """Export 29 Fourier-informed interior points and Wolfram model data."""
    model = XSMThermalModel()
    fourier = optimize_path(temperature, output_directory)
    point_set = prepare_findbounce_points(
        fourier,
        DEFAULT_FIND_BOUNCE_INTERIOR_POINTS,
        sampling="arclength",
    )
    files = export_findbounce_points(
        point_set,
        output_directory / "findbounce_input",
        basename="xsm_initializer",
    )
    _write_json(
        output_directory / "xsm_wolfram_parameters.json",
        model.wolfram_definitions(temperature),
    )
    return files["metadata"]


def run_findbounce(temperature: float, output_directory: Path) -> dict[str, Any]:
    """Export the path and invoke the genuine Wolfram/FindBounce driver."""
    export_findbounce(temperature, output_directory)
    executable = shutil.which("wolframscript")
    if executable is None:
        raise RuntimeError(
            "wolframscript is unavailable; run the findbounce-export stage and "
            "consume its geometry in a Mathematica installation with FindBounce"
        )
    script = Path(__file__).with_name("findbounce.wls")
    wrapper = (
        Path(__file__).resolve().parents[2]
        / "fourier_path_bounce"
        / "wolfram"
        / "FourierPathBounce.wl"
    )
    environment = os.environ.copy()
    environment.update(
        XSM_EXAMPLE_OUTPUT=str(output_directory.resolve()),
        FOURIER_PATH_BOUNCE_WOLFRAM=str(wrapper),
    )
    process = subprocess.run(
        [executable, "-file", str(script)],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
        timeout=330,
    )
    (output_directory / "findbounce.log").write_text(
        process.stdout + process.stderr, encoding="utf-8"
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Wolfram/FindBounce failed with exit code {process.returncode}; "
            f"see {output_directory / 'findbounce.log'}"
        )
    result_file = output_directory / "findbounce_results.json"
    if not result_file.is_file():
        raise RuntimeError("FindBounce did not create findbounce_results.json")
    return json.loads(result_file.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "validate",
            "fourier",
            "cosmotransitions",
            "findbounce-export",
            "findbounce",
            "all",
        ),
        help="calculation stage to execute",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE_GEV,
        help="temperature in GeV (default: 85)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("xsm_example_output"),
        help="directory for generated files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_directory = args.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)
    if args.stage in ("validate", "all"):
        result = validate_model(args.temperature, output_directory)
        print(json.dumps(result, indent=2, sort_keys=True))
    if args.stage in ("fourier", "all"):
        result = optimize_path(args.temperature, output_directory)
        print(json.dumps(result.summary(), indent=2, sort_keys=True, default=_json_default))
    if args.stage in ("cosmotransitions", "all"):
        print(json.dumps(run_ct(args.temperature, output_directory), indent=2, sort_keys=True))
    if args.stage == "findbounce-export":
        print(export_findbounce(args.temperature, output_directory))
    if args.stage in ("findbounce", "all"):
        print(json.dumps(run_findbounce(args.temperature, output_directory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
