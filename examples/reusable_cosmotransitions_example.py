"""Run the final CosmoTransitions solver from a Fourier initializer."""

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, gradient, potential
from fourier_path_bounce import (
    CosmoTransitionsSettings,
    FourierPathSettings,
    ModeSelectionSettings,
    optimize_fourier_path,
    run_cosmotransitions,
)


fourier = optimize_fourier_path(
    potential,
    FALSE_VACUUM,
    TRUE_VACUUM,
    settings=FourierPathSettings(
        dimension=4,
        n_grid=130,
        mode_selection=ModeSelectionSettings(modes=(1, 2), patience=2),
    ),
)
result = run_cosmotransitions(
    potential,
    gradient,
    FALSE_VACUUM,
    TRUE_VACUUM,
    fourier_result=fourier,
    n_path_points=80,
    settings=CosmoTransitionsSettings(maxiter=40),
)
print(result.summary())
