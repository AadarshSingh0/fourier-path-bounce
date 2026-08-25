"""Command-line interface for reusable Fourier preconditioning."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .core import (
    FourierPathError,
    FourierPathSettings,
    InputValidationError,
    ModeSelectionSettings,
    OptimizerSettings,
    optimize_fourier_path,
)
from .cosmotransitions import CosmoTransitionsSettings, prepare_cosmotransitions_path, run_cosmotransitions
from .findbounce import export_findbounce_points, prepare_findbounce_points
from .serialization import load_fourier_result, save_fourier_result


def _callable(specification: str) -> Callable[[Any], Any]:
    if ":" not in specification:
        raise InputValidationError("callables must use module:function syntax")
    module_name, attribute = specification.split(":", 1)
    if not module_name or not attribute or any(part.startswith("_") for part in attribute.split(".")):
        raise InputValidationError("invalid module:function specification")
    try:
        value: Any = importlib.import_module(module_name)
        for part in attribute.split("."):
            value = getattr(value, part)
    except Exception as exc:
        raise InputValidationError(f"cannot load {specification}: {exc}") from exc
    if not callable(value):
        raise InputValidationError(f"{specification} is not callable")
    return value


def _vector(text: str) -> np.ndarray:
    try:
        result = np.asarray([float(item.strip()) for item in text.split(",")], dtype=float)
    except Exception as exc:
        raise argparse.ArgumentTypeError("expected comma-separated finite numbers") from exc
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError("expected comma-separated finite numbers")
    return result


def _modes(text: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except Exception as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive mode counts") from exc
    if not result or tuple(sorted(set(result))) != result or result[0] < 1:
        raise argparse.ArgumentTypeError("modes must be increasing unique positive integers")
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def _add_result_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--result", required=True, type=Path, help="saved Fourier .npz result")


def _optimization_settings(arguments: argparse.Namespace) -> FourierPathSettings:
    return FourierPathSettings(
        dimension=arguments.dimension,
        n_grid=arguments.n_grid,
        optimizer=OptimizerSettings(
            maxiter=arguments.maxiter,
            ftol=arguments.ftol,
            gtol=arguments.gtol,
            maxls=arguments.maxls,
            coefficient_bound_factor=arguments.bound_factor,
        ),
        mode_selection=ModeSelectionSettings(
            modes=arguments.modes,
            relative_tolerance=arguments.mode_tolerance,
            patience=arguments.patience,
            zero_start=True,
            warm_previous=True,
            warm_best=arguments.warm_best,
            random_starts=arguments.random_starts,
            random_scale=arguments.random_scale,
            seed=arguments.seed,
        ),
    )


def _cmd_optimize(arguments: argparse.Namespace) -> int:
    potential = _callable(arguments.potential)
    gradient = _callable(arguments.gradient) if arguments.gradient else None
    result = optimize_fourier_path(
        potential,
        arguments.false,
        arguments.true,
        settings=_optimization_settings(arguments),
        potential_gradient=gradient,
        metadata={"potential_spec": arguments.potential, "gradient_spec": arguments.gradient},
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_fourier_result(result, arguments.output_dir / "fourier_result.npz")
    _write_json(arguments.output_dir / "fourier_summary.json", result.summary())
    print(result_path)
    return 0


def _cmd_sample(arguments: argparse.Namespace) -> int:
    result = load_fourier_result(arguments.result)
    path = result.sample(arguments.points, sampling=arguments.sampling)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(arguments.output, path, delimiter=",", fmt="%.17g")
    print(arguments.output)
    return 0


def _cmd_ct_prepare(arguments: argparse.Namespace) -> int:
    result = load_fourier_result(arguments.result)
    path = prepare_cosmotransitions_path(
        result.false_vacuum,
        result.true_vacuum,
        fourier_result=result,
        initialization=arguments.initialization,
        n_points=arguments.points,
        sampling=arguments.sampling,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = arguments.output_dir / "cosmotransitions_path_true_to_false.csv"
    np.savetxt(csv_path, path, delimiter=",", fmt="%.17g")
    _write_json(
        arguments.output_dir / "cosmotransitions_path_metadata.json",
        {
            "format": "fourier_path_bounce.cosmotransitions_points",
            "initialization": arguments.initialization,
            "public_orientation": "false_to_true",
            "file_orientation": "true_to_false",
            "path_shape": list(path.shape),
            "true_vacuum": result.true_vacuum.tolist(),
            "false_vacuum": result.false_vacuum.tolist(),
            "sampling": arguments.sampling,
            "selected_fourier_modes": result.selected_modes,
        },
    )
    print(csv_path)
    return 0


def _cmd_findbounce_export(arguments: argparse.Namespace) -> int:
    result = load_fourier_result(arguments.result)
    points = prepare_findbounce_points(result, arguments.K, sampling=arguments.sampling)
    files = export_findbounce_points(points, arguments.output_dir, basename=arguments.basename)
    print(files["metadata"])
    return 0


def _cmd_ct_run(arguments: argparse.Namespace) -> int:
    potential = _callable(arguments.potential)
    gradient = _callable(arguments.gradient)
    stored = load_fourier_result(arguments.result) if arguments.result else None
    if stored is not None:
        false, true = stored.false_vacuum, stored.true_vacuum
    else:
        if arguments.false is None or arguments.true is None:
            raise InputValidationError("--false and --true are required when --result is absent")
        false, true = arguments.false, arguments.true
    result = run_cosmotransitions(
        potential,
        gradient,
        false,
        true,
        fourier_result=stored,
        fourier_settings=None if stored is not None else FourierPathSettings(),
        initialization=arguments.initialization,
        n_path_points=arguments.points,
        sampling=arguments.sampling,
        settings=CosmoTransitionsSettings(dimension=arguments.dimension, maxiter=arguments.ct_maxiter),
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output_dir / "cosmotransitions_result.json", result.summary())
    np.savez_compressed(
        arguments.output_dir / "cosmotransitions_arrays.npz",
        initial_path_false_to_true=result.initial_path_false_to_true,
        initial_path_cosmotransitions=result.initial_path_cosmotransitions,
        final_path=result.final_path,
        radial_grid=result.radial_grid,
        profile_coordinate=result.profile_coordinate,
        profile_derivative=result.profile_derivative,
    )
    (arguments.output_dir / "cosmotransitions.log").write_text(result.solver_log, encoding="ascii")
    print(arguments.output_dir / "cosmotransitions_result.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fourier-path-bounce",
        description="Optimize and export reusable Fourier-preconditioned tunnelling paths.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser("optimize", help="optimize and save a Fourier path")
    optimize.add_argument("--potential", required=True, help="JAX-compatible module:function")
    optimize.add_argument("--gradient", help="optional module:function gradient")
    optimize.add_argument("--false", required=True, type=_vector, help="comma-separated false vacuum")
    optimize.add_argument("--true", required=True, type=_vector, help="comma-separated true vacuum")
    optimize.add_argument("--dimension", type=int, choices=(3, 4), default=4)
    optimize.add_argument("--n-grid", type=int, default=260)
    optimize.add_argument("--modes", type=_modes, default=(1, 2, 3, 4, 5))
    optimize.add_argument("--mode-tolerance", type=float, default=1.0e-3)
    optimize.add_argument("--patience", type=int, default=3)
    optimize.add_argument("--maxiter", type=int, default=800)
    optimize.add_argument("--ftol", type=float, default=1.0e-10)
    optimize.add_argument("--gtol", type=float, default=1.0e-7)
    optimize.add_argument("--maxls", type=int, default=50)
    optimize.add_argument("--bound-factor", type=float, default=0.5)
    optimize.add_argument("--warm-best", action="store_true")
    optimize.add_argument("--random-starts", type=int, default=0)
    optimize.add_argument("--random-scale", type=float, default=0.03)
    optimize.add_argument("--seed", type=int, default=20260508)
    optimize.add_argument("--output-dir", required=True, type=Path)
    optimize.set_defaults(handler=_cmd_optimize)

    sample = subparsers.add_parser("sample", help="sample a saved Fourier path")
    _add_result_argument(sample)
    sample.add_argument("--points", required=True, type=int)
    sample.add_argument("--sampling", choices=("parameter", "arclength"), default="parameter")
    sample.add_argument("--output", required=True, type=Path)
    sample.set_defaults(handler=_cmd_sample)

    ct_prepare = subparsers.add_parser("ct-prepare", help="write CosmoTransitions-oriented path points")
    _add_result_argument(ct_prepare)
    ct_prepare.add_argument("--initialization", choices=("fourier", "straight"), default="fourier")
    ct_prepare.add_argument("--points", type=int, default=120)
    ct_prepare.add_argument("--sampling", choices=("parameter", "arclength"), default="parameter")
    ct_prepare.add_argument("--output-dir", required=True, type=Path)
    ct_prepare.set_defaults(handler=_cmd_ct_prepare)

    findbounce = subparsers.add_parser("findbounce-export", help="export K FindBounce interior points")
    _add_result_argument(findbounce)
    findbounce.add_argument("--K", required=True, type=int, help="number of interior points, not N_m")
    findbounce.add_argument("--sampling", choices=("parameter", "arclength"), default="parameter")
    findbounce.add_argument("--basename", default="findbounce_fourier")
    findbounce.add_argument("--output-dir", required=True, type=Path)
    findbounce.set_defaults(handler=_cmd_findbounce_export)

    ct_run = subparsers.add_parser("ct-run", help="run standard CosmoTransitions path deformation")
    ct_run.add_argument("--potential", required=True, help="module:function")
    ct_run.add_argument("--gradient", required=True, help="module:function")
    ct_run.add_argument("--result", type=Path, help="saved Fourier result")
    ct_run.add_argument("--false", type=_vector, help="required without --result")
    ct_run.add_argument("--true", type=_vector, help="required without --result")
    ct_run.add_argument("--initialization", choices=("fourier", "straight"), default="fourier")
    ct_run.add_argument("--points", type=int, default=120)
    ct_run.add_argument("--sampling", choices=("parameter", "arclength"), default="parameter")
    ct_run.add_argument("--dimension", type=int, choices=(3, 4), default=4)
    ct_run.add_argument("--ct-maxiter", type=int, default=40)
    ct_run.add_argument("--output-dir", required=True, type=Path)
    ct_run.set_defaults(handler=_cmd_ct_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        raw_arguments = list(sys.argv[1:] if argv is None else argv)
        # argparse otherwise mistakes a comma-separated vector beginning with a
        # negative number for a new option. Both --false=-1,0 and --false -1,0
        # are accepted by normalizing the latter before parsing.
        for flag in ("--false", "--true"):
            if flag in raw_arguments:
                index = raw_arguments.index(flag)
                if index + 1 < len(raw_arguments) and raw_arguments[index + 1].startswith("-"):
                    raw_arguments[index : index + 2] = [f"{flag}={raw_arguments[index + 1]}"]
        arguments = parser.parse_args(raw_arguments)
        return int(arguments.handler(arguments))
    except (FourierPathError, InputValidationError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
