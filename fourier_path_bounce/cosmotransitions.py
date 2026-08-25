"""Reusable adapter around CosmoTransitions' standard path deformation."""

from __future__ import annotations

import contextlib
import io
import re
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .core import (
    FourierPathError,
    FourierPathResult,
    FourierPathSettings,
    InputValidationError,
    PotentialEvaluationError,
    optimize_fourier_path,
)


@dataclass(frozen=True)
class CosmoTransitionsSettings:
    dimension: int = 4
    maxiter: int = 40
    fix_end_cutoff: float = 0.03
    verbose: bool = False
    v_spline_samples: int = 100
    tunneling_init_params: Mapping[str, Any] = field(default_factory=dict)
    tunneling_find_profile_params: Mapping[str, Any] = field(default_factory=dict)
    deformation_init_params: Mapping[str, Any] = field(default_factory=dict)
    deformation_deform_params: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dimension not in (3, 4):
            raise InputValidationError("CosmoTransitions dimension must be 3 or 4")
        if self.maxiter < 1 or self.v_spline_samples < 10:
            raise InputValidationError("CosmoTransitions maxiter and spline samples are too small")
        if not 0.0 <= self.fix_end_cutoff < 1.0:
            raise InputValidationError("fix_end_cutoff must lie in [0, 1)")


@dataclass
class CosmoTransitionsResult:
    initialization: str
    initial_path_false_to_true: np.ndarray
    initial_path_cosmotransitions: np.ndarray
    final_path: np.ndarray
    radial_grid: np.ndarray
    profile_coordinate: np.ndarray
    profile_derivative: np.ndarray
    action: float
    f_ratio: float
    deformation_steps: int | None
    deformation_rounds: int | None
    outer_iteration_limit_reached: bool
    solver_success: bool
    status: str
    warnings: tuple[str, ...]
    solver_log: str
    fourier_time_s: float
    solver_time_s: float
    total_time_s: float
    fourier_result: FourierPathResult | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "initialization": self.initialization,
            "action": self.action,
            "f_ratio": self.f_ratio,
            "deformation_steps": self.deformation_steps,
            "deformation_rounds": self.deformation_rounds,
            "outer_iteration_limit_reached": self.outer_iteration_limit_reached,
            "solver_success": self.solver_success,
            "status": self.status,
            "warnings": list(self.warnings),
            "fourier_time_s": self.fourier_time_s,
            "solver_time_s": self.solver_time_s,
            "total_time_s": self.total_time_s,
            "initial_path_shape": list(self.initial_path_false_to_true.shape),
            "final_path_shape": list(self.final_path.shape),
            "metadata": self.metadata,
        }


def _endpoints(false_vacuum: Sequence[float], true_vacuum: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    false = np.asarray(false_vacuum, dtype=float)
    true = np.asarray(true_vacuum, dtype=float)
    if false.ndim != 1 or true.ndim != 1 or false.size == 0 or false.shape != true.shape:
        raise InputValidationError("CosmoTransitions vacua must be matching one-dimensional arrays")
    if not np.all(np.isfinite(false)) or not np.all(np.isfinite(true)):
        raise InputValidationError("CosmoTransitions vacua must be finite")
    return false, true


def prepare_cosmotransitions_path(
    false_vacuum: Sequence[float],
    true_vacuum: Sequence[float],
    *,
    fourier_result: FourierPathResult | None = None,
    initialization: str = "fourier",
    n_points: int = 120,
    sampling: str = "parameter",
) -> np.ndarray:
    """Return the true-to-false point array required by CosmoTransitions 2.x."""
    false, true = _endpoints(false_vacuum, true_vacuum)
    if not isinstance(n_points, int) or n_points < 3:
        raise InputValidationError("CosmoTransitions n_points must be an integer >= 3")
    if initialization == "straight":
        path_false_to_true = np.linspace(false, true, n_points)
    elif initialization == "fourier":
        if fourier_result is None:
            raise InputValidationError("fourier initialization requires a FourierPathResult")
        if not np.array_equal(fourier_result.false_vacuum, false) or not np.array_equal(
            fourier_result.true_vacuum, true
        ):
            raise InputValidationError("Fourier result endpoints do not match CosmoTransitions vacua")
        path_false_to_true = fourier_result.sample(n_points, sampling=sampling)
    else:
        raise InputValidationError("initialization must be 'fourier' or 'straight'")
    path_false_to_true[0] = false
    path_false_to_true[-1] = true
    # fullTunneling requires lower/true first and metastable/false last.
    path_ct = path_false_to_true[::-1].copy()
    path_ct[0] = true
    path_ct[-1] = false
    return path_ct


def _parse_steps(log: str) -> tuple[int | None, int | None, float | None]:
    matches = re.findall(
        r"Path deformation converged\.\s*(\d+)\s*steps\.\s*fRatio\s*=\s*([0-9.eE+-]+)",
        log,
    )
    if not matches:
        return None, None, None
    return sum(int(item[0]) for item in matches), len(matches), float(matches[-1][1])


def _evaluate_user_function(function: Callable[[Any], Any], points: Any, *, gradient: bool) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    one_point = array.ndim == 1
    batch = array.reshape(1, -1) if one_point else array
    outputs = []
    for index, point in enumerate(batch):
        try:
            value = np.asarray(function(point), dtype=float)
        except Exception as exc:
            kind = "gradient" if gradient else "potential"
            raise PotentialEvaluationError(f"{kind} failed at solver point {index}: {exc}") from exc
        expected = point.shape if gradient else ()
        if value.shape != expected or not np.all(np.isfinite(value)):
            kind = "gradient" if gradient else "potential"
            raise PotentialEvaluationError(
                f"{kind} returned shape {value.shape}, expected {expected}, or a non-finite value"
            )
        outputs.append(value)
    result = np.asarray(outputs, dtype=float)
    return result[0] if one_point else result


def run_cosmotransitions(
    potential: Callable[[Any], Any],
    gradient: Callable[[Any], Any],
    false_vacuum: Sequence[float],
    true_vacuum: Sequence[float],
    *,
    fourier_result: FourierPathResult | None = None,
    fourier_settings: FourierPathSettings | None = None,
    initialization: str = "fourier",
    n_path_points: int = 120,
    sampling: str = "parameter",
    settings: CosmoTransitionsSettings | None = None,
) -> CosmoTransitionsResult:
    """Run standard CosmoTransitions path deformation from a selected initializer."""
    started = time.perf_counter()
    false, true = _endpoints(false_vacuum, true_vacuum)
    ct_settings = settings or CosmoTransitionsSettings()
    ct_settings.validate()
    fourier_time = 0.0
    generated_result = fourier_result
    if initialization == "fourier" and generated_result is None:
        preprocess_started = time.perf_counter()
        generated_result = optimize_fourier_path(
            potential,
            false,
            true,
            settings=fourier_settings or FourierPathSettings(dimension=ct_settings.dimension),
            potential_gradient=gradient,
        )
        fourier_time = time.perf_counter() - preprocess_started
    elif generated_result is not None:
        fourier_time = generated_result.total_time_s
    if generated_result is not None and generated_result.settings.dimension != ct_settings.dimension:
        raise InputValidationError(
            "Fourier and CosmoTransitions Euclidean dimensions must match"
        )
    path_ct = prepare_cosmotransitions_path(
        false,
        true,
        fourier_result=generated_result,
        initialization=initialization,
        n_points=n_path_points,
        sampling=sampling,
    )
    initial_false_to_true = path_ct[::-1].copy()

    try:
        from cosmoTransitions import pathDeformation
    except Exception as exc:
        raise FourierPathError(
            "CosmoTransitions is unavailable; install a compatible 2.x release to run this adapter"
        ) from exc

    def solver_potential(points: Any) -> np.ndarray:
        return _evaluate_user_function(potential, points, gradient=False)

    def solver_gradient(points: Any) -> np.ndarray:
        return _evaluate_user_function(gradient, points, gradient=True)

    captured = io.StringIO()
    caught: list[warnings.WarningMessage] = []
    solver_started = time.perf_counter()
    try:
        tunneling_init_params = dict(ct_settings.tunneling_init_params)
        expected_alpha = ct_settings.dimension - 1
        if "alpha" in tunneling_init_params and tunneling_init_params["alpha"] != expected_alpha:
            raise InputValidationError(
                "tunneling_init_params['alpha'] conflicts with the requested Euclidean dimension"
            )
        tunneling_init_params["alpha"] = expected_alpha
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                output = pathDeformation.fullTunneling(
                    path_ct,
                    solver_potential,
                    solver_gradient,
                    maxiter=ct_settings.maxiter,
                    fixEndCutoff=ct_settings.fix_end_cutoff,
                    verbose=ct_settings.verbose,
                    V_spline_samples=ct_settings.v_spline_samples,
                    tunneling_init_params=tunneling_init_params,
                    tunneling_findProfile_params=dict(ct_settings.tunneling_find_profile_params),
                    deformation_init_params=dict(ct_settings.deformation_init_params),
                    deformation_deform_params=dict(ct_settings.deformation_deform_params),
                )
    except Exception as exc:
        detail = captured.getvalue()
        raise FourierPathError(
            f"CosmoTransitions fullTunneling failed: {exc}\n{detail}\n{traceback.format_exc()}"
        ) from exc
    solver_time = time.perf_counter() - solver_started
    log = captured.getvalue()
    steps, rounds, parsed_ratio = _parse_steps(log)
    action = float(output.action)
    f_ratio = float(output.fRatio) if np.isfinite(float(output.fRatio)) else float(parsed_ratio)
    final_path = np.asarray(output.Phi, dtype=float)
    radial = np.asarray(output.profile1D.R, dtype=float)
    profile_coordinate = np.asarray(output.profile1D.Phi, dtype=float)
    profile_derivative = np.asarray(output.profile1D.dPhi, dtype=float)
    arrays = (final_path, radial, profile_coordinate, profile_derivative)
    if not np.isfinite(action) or not np.isfinite(f_ratio) or not all(
        np.all(np.isfinite(array)) for array in arrays
    ):
        raise FourierPathError("CosmoTransitions returned non-finite output")
    adapter_wall_time = time.perf_counter() - started
    total_time = fourier_time + solver_time
    outer_limit_reached = rounds is not None and rounds >= ct_settings.maxiter
    return CosmoTransitionsResult(
        initialization=initialization,
        initial_path_false_to_true=initial_false_to_true,
        initial_path_cosmotransitions=path_ct,
        final_path=final_path,
        radial_grid=radial,
        profile_coordinate=profile_coordinate,
        profile_derivative=profile_derivative,
        action=action,
        f_ratio=f_ratio,
        deformation_steps=steps,
        deformation_rounds=rounds,
        outer_iteration_limit_reached=outer_limit_reached,
        solver_success=True,
        status="returned_at_outer_iteration_limit" if outer_limit_reached else "ok",
        warnings=tuple(str(item.message) for item in caught),
        solver_log=log,
        fourier_time_s=fourier_time,
        solver_time_s=solver_time,
        total_time_s=total_time,
        fourier_result=generated_result,
        metadata={
            "adapter": "cosmoTransitions.pathDeformation.fullTunneling",
            "public_orientation": "false_to_true",
            "solver_input_orientation": "true_to_false",
            "sampling": sampling,
            "n_path_points": n_path_points,
            "dimension": ct_settings.dimension,
            "cosmotransitions_alpha": ct_settings.dimension - 1,
            "adapter_wall_time_this_call_s": adapter_wall_time,
        },
    )


__all__ = [
    "CosmoTransitionsResult",
    "CosmoTransitionsSettings",
    "prepare_cosmotransitions_path",
    "run_cosmotransitions",
]
