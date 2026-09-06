"""Portable serialization for Fourier path results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    FourierPathResult,
    FourierPathSettings,
    InputValidationError,
    ModeResult,
    ModeSelectionSettings,
    OptimizerSettings,
)


FORMAT_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_fourier_result(result: FourierPathResult, path: str | Path) -> Path:
    """Save a result as one compressed NPZ with embedded JSON metadata."""
    destination = Path(path)
    if destination.suffix != ".npz":
        raise InputValidationError("Fourier result filenames must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "fourier_path_bounce.result",
        "format_version": FORMAT_VERSION,
        "selected_modes": result.selected_modes,
        "action_proxy": result.action_proxy,
        "optimizer_success": result.optimizer_success,
        "optimizer_message": result.optimizer_message,
        "adaptive_converged": result.adaptive_converged,
        "stop_reason": result.stop_reason,
        "settings": asdict(result.settings),
        "potential_false": result.potential_false,
        "potential_true": result.potential_true,
        "compile_time_s": result.compile_time_s,
        "solve_time_s": result.solve_time_s,
        "total_time_s": result.total_time_s,
        "gradient_norm": result.gradient_norm,
        "coefficient_norm": result.coefficient_norm,
        "coefficient_bound": result.coefficient_bound,
        "coefficient_bound_saturation_count": result.coefficient_bound_saturation_count,
        "invalid_interval_count": result.invalid_interval_count,
        "min_v_minus_vt": result.min_v_minus_vt,
        "optimizer_made_no_progress": result.optimizer_made_no_progress,
        "history": [asdict(item) for item in result.history],
        "metadata": result.metadata,
        "orientation": result.orientation,
        "path_sha256": result.path_sha256,
    }
    np.savez_compressed(
        destination,
        false_vacuum=np.asarray(result.false_vacuum, dtype=float),
        true_vacuum=np.asarray(result.true_vacuum, dtype=float),
        coefficients=np.asarray(result.coefficients, dtype=float),
        parameter=np.asarray(result.parameter, dtype=float),
        path_points=np.asarray(result.path_points, dtype=float),
        metadata_json=np.asarray(json.dumps(payload, default=_json_default, sort_keys=True)),
    )
    return destination


def load_fourier_result(path: str | Path) -> FourierPathResult:
    """Load and validate a result produced by :func:`save_fourier_result`."""
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            payload = json.loads(str(archive["metadata_json"].item()))
            arrays = {name: np.asarray(archive[name], dtype=float) for name in (
                "false_vacuum", "true_vacuum", "coefficients", "parameter", "path_points"
            )}
    except Exception as exc:
        raise InputValidationError(f"cannot load Fourier result {source}: {exc}") from exc
    if payload.get("format") != "fourier_path_bounce.result" or payload.get("format_version") != FORMAT_VERSION:
        raise InputValidationError("unsupported Fourier result format")
    if payload.get("orientation") != "false_to_true":
        raise InputValidationError("serialized path is not marked false_to_true")
    settings_data = payload["settings"]
    settings = FourierPathSettings(
        dimension=int(settings_data["dimension"]),
        n_grid=int(settings_data["n_grid"]),
        profile=str(settings_data["profile"]),
        optimizer=OptimizerSettings(**settings_data["optimizer"]),
        mode_selection=ModeSelectionSettings(
            **{
                **settings_data["mode_selection"],
                "modes": tuple(settings_data["mode_selection"]["modes"]),
            }
        ),
    )
    history = tuple(ModeResult(**item) for item in payload["history"])
    result = FourierPathResult(
        false_vacuum=arrays["false_vacuum"],
        true_vacuum=arrays["true_vacuum"],
        coefficients=arrays["coefficients"],
        selected_modes=int(payload["selected_modes"]),
        parameter=arrays["parameter"],
        path_points=arrays["path_points"],
        action_proxy=float(payload["action_proxy"]),
        optimizer_success=bool(payload["optimizer_success"]),
        optimizer_message=str(payload["optimizer_message"]),
        adaptive_converged=bool(payload["adaptive_converged"]),
        stop_reason=str(payload["stop_reason"]),
        history=history,
        settings=settings,
        potential_false=float(payload["potential_false"]),
        potential_true=float(payload["potential_true"]),
        compile_time_s=float(payload["compile_time_s"]),
        solve_time_s=float(payload["solve_time_s"]),
        total_time_s=float(payload["total_time_s"]),
        gradient_norm=float(payload["gradient_norm"]),
        coefficient_norm=float(payload["coefficient_norm"]),
        coefficient_bound=float(payload["coefficient_bound"]),
        coefficient_bound_saturation_count=int(payload["coefficient_bound_saturation_count"]),
        invalid_interval_count=int(payload["invalid_interval_count"]),
        min_v_minus_vt=float(payload["min_v_minus_vt"]),
        # Absent in v0.1.0 archives; those runs predate the diagnostic.
        optimizer_made_no_progress=bool(payload.get("optimizer_made_no_progress", False)),
        metadata=dict(payload.get("metadata", {})),
    )
    if result.path_points.shape != (result.parameter.size, result.n_fields):
        raise InputValidationError("serialized path shape is inconsistent")
    if result.coefficients.shape != (result.n_fields, result.selected_modes):
        raise InputValidationError("serialized coefficient shape is inconsistent")
    if not np.array_equal(result.path_points[0], result.false_vacuum):
        raise InputValidationError("serialized path does not preserve the false endpoint")
    if not np.array_equal(result.path_points[-1], result.true_vacuum):
        raise InputValidationError("serialized path does not preserve the true endpoint")
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise InputValidationError("serialized result contains non-finite arrays")
    if result.path_sha256 != payload.get("path_sha256"):
        raise InputValidationError("serialized path checksum mismatch")
    return result


__all__ = ["FORMAT_VERSION", "load_fourier_result", "save_fourier_result"]
