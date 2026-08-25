"""FindBounce geometry export with an explicit Python/Wolfram boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import FourierPathResult, InputValidationError


EXPORT_FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FindBouncePointSet:
    """Open false-to-true polygonal initializer for FindBounce."""

    false_vacuum: np.ndarray
    true_vacuum: np.ndarray
    interior_points: np.ndarray
    full_path: np.ndarray
    K: int
    selected_fourier_modes: int
    sampling: str
    source_path_sha256: str

    @property
    def n_fields(self) -> int:
        return int(self.false_vacuum.size)

    def metadata(self) -> dict[str, Any]:
        return {
            "format": "fourier_path_bounce.findbounce_points",
            "format_version": EXPORT_FORMAT_VERSION,
            "orientation": "false_to_true",
            "polygon": "open",
            "n_fields": self.n_fields,
            "K": self.K,
            "selected_fourier_modes": self.selected_fourier_modes,
            "sampling": self.sampling,
            "false_vacuum": self.false_vacuum.tolist(),
            "true_vacuum": self.true_vacuum.tolist(),
            "interior_shape": list(self.interior_points.shape),
            "full_path_shape": list(self.full_path.shape),
            "source_path_sha256": self.source_path_sha256,
        }


def prepare_findbounce_points(
    result: FourierPathResult,
    K: int,
    *,
    sampling: str = "parameter",
) -> FindBouncePointSet:
    """Sample ``K`` interior points; ``K`` is independent of Fourier ``N_m``."""
    if not isinstance(K, int) or isinstance(K, bool) or K < 1:
        raise InputValidationError("K must be an integer >= 1")
    if sampling not in ("parameter", "arclength"):
        raise InputValidationError("sampling must be 'parameter' or 'arclength'")
    full_path = result.sample(K + 2, sampling=sampling)
    interior = full_path[1:-1].copy()
    if full_path.shape != (K + 2, result.n_fields) or interior.shape != (K, result.n_fields):
        raise InputValidationError("sampled FindBounce point dimensions are inconsistent")
    if not np.all(np.isfinite(full_path)):
        raise InputValidationError("FindBounce points must be finite")
    if not np.array_equal(full_path[0], result.false_vacuum) or not np.array_equal(
        full_path[-1], result.true_vacuum
    ):
        raise InputValidationError("FindBounce path endpoints are not preserved")
    if np.any(np.all(interior == result.false_vacuum, axis=1)) or np.any(
        np.all(interior == result.true_vacuum, axis=1)
    ):
        raise InputValidationError("FindBounce interior points duplicate an endpoint")
    return FindBouncePointSet(
        false_vacuum=result.false_vacuum.copy(),
        true_vacuum=result.true_vacuum.copy(),
        interior_points=interior,
        full_path=full_path,
        K=K,
        selected_fourier_modes=result.selected_modes,
        sampling=sampling,
        source_path_sha256=result.path_sha256,
    )


def export_findbounce_points(
    point_set: FindBouncePointSet,
    output_directory: str | Path,
    *,
    basename: str = "findbounce_fourier",
) -> dict[str, Path]:
    """Export full-precision CSV geometry and a self-describing JSON file."""
    if not basename or Path(basename).name != basename:
        raise InputValidationError("basename must be one simple filename component")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    interior_path = directory / f"{basename}_interior.csv"
    full_path = directory / f"{basename}_open_path.csv"
    metadata_path = directory / f"{basename}_metadata.json"
    np.savetxt(interior_path, point_set.interior_points, delimiter=",", fmt="%.17g")
    np.savetxt(full_path, point_set.full_path, delimiter=",", fmt="%.17g")
    metadata = point_set.metadata()
    metadata.update(
        {
            "interior_csv": interior_path.name,
            "full_path_csv": full_path.name,
            "interior_csv_sha256": _sha256(interior_path),
            "full_path_csv_sha256": _sha256(full_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return {"metadata": metadata_path, "interior_csv": interior_path, "full_path_csv": full_path}


def load_findbounce_points(metadata_file: str | Path) -> FindBouncePointSet:
    """Round-trip a FindBounce export and verify its content hashes."""
    metadata_path = Path(metadata_file)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    except Exception as exc:
        raise InputValidationError(f"cannot read FindBounce metadata: {exc}") from exc
    if metadata.get("format") != "fourier_path_bounce.findbounce_points":
        raise InputValidationError("unsupported FindBounce export format")
    if metadata.get("format_version") != EXPORT_FORMAT_VERSION:
        raise InputValidationError("unsupported FindBounce export version")
    if metadata.get("orientation") != "false_to_true" or metadata.get("polygon") != "open":
        raise InputValidationError("FindBounce export must be an open false-to-true path")
    interior_path = metadata_path.parent / metadata["interior_csv"]
    full_path_file = metadata_path.parent / metadata["full_path_csv"]
    if _sha256(interior_path) != metadata["interior_csv_sha256"] or _sha256(
        full_path_file
    ) != metadata["full_path_csv_sha256"]:
        raise InputValidationError("FindBounce CSV checksum mismatch")
    n_fields = int(metadata["n_fields"])
    K = int(metadata["K"])
    interior = np.loadtxt(interior_path, delimiter=",", ndmin=2)
    full_path = np.loadtxt(full_path_file, delimiter=",", ndmin=2)
    false = np.asarray(metadata["false_vacuum"], dtype=float)
    true = np.asarray(metadata["true_vacuum"], dtype=float)
    if interior.shape != (K, n_fields) or full_path.shape != (K + 2, n_fields):
        raise InputValidationError("FindBounce CSV shape does not match metadata")
    if not np.array_equal(full_path[0], false) or not np.array_equal(full_path[-1], true):
        raise InputValidationError("FindBounce CSV endpoint orientation is inconsistent")
    return FindBouncePointSet(
        false_vacuum=false,
        true_vacuum=true,
        interior_points=interior,
        full_path=full_path,
        K=K,
        selected_fourier_modes=int(metadata["selected_fourier_modes"]),
        sampling=str(metadata["sampling"]),
        source_path_sha256=str(metadata["source_path_sha256"]),
    )


__all__ = [
    "EXPORT_FORMAT_VERSION",
    "FindBouncePointSet",
    "export_findbounce_points",
    "load_findbounce_points",
    "prepare_findbounce_points",
]
