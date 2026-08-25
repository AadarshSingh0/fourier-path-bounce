"""Export K=1 and K=4 open FindBounce initial paths."""

from pathlib import Path

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, potential
from fourier_path_bounce import (
    FourierPathSettings,
    ModeSelectionSettings,
    export_findbounce_points,
    optimize_fourier_path,
    prepare_findbounce_points,
)


result = optimize_fourier_path(
    potential,
    FALSE_VACUUM,
    TRUE_VACUUM,
    settings=FourierPathSettings(
        n_grid=130,
        mode_selection=ModeSelectionSettings(modes=(1, 2), patience=2),
    ),
)
for k in (1, 4):
    point_set = prepare_findbounce_points(result, k, sampling="arclength")
    files = export_findbounce_points(
        point_set, Path("example_output") / f"findbounce_K{k}", basename="initializer"
    )
    print(files["metadata"])
