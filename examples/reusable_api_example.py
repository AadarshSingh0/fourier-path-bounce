"""Optimize and serialize a new user-supplied two-field path."""

from pathlib import Path

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, potential
from fourier_path_bounce import (
    FourierPathSettings,
    ModeSelectionSettings,
    optimize_fourier_path,
    save_fourier_result,
)


settings = FourierPathSettings(
    dimension=4,
    n_grid=130,
    mode_selection=ModeSelectionSettings(modes=(1, 2, 3), patience=2),
)
result = optimize_fourier_path(potential, FALSE_VACUUM, TRUE_VACUUM, settings=settings)
destination = save_fourier_result(result, Path("example_output") / "fourier_result.npz")
print(destination)
print(result.summary())
