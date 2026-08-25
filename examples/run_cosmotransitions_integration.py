"""Auditable real CosmoTransitions run on the established curved-valley case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.curved_valley_potential import FALSE_VACUUM, TRUE_VACUUM, gradient, potential
from fourier_path_bounce import (
    CosmoTransitionsSettings,
    FourierPathSettings,
    ModeSelectionSettings,
    OptimizerSettings,
    optimize_fourier_path,
    run_cosmotransitions,
    save_fourier_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    fourier = optimize_fourier_path(
        potential,
        FALSE_VACUUM,
        TRUE_VACUUM,
        settings=FourierPathSettings(
            dimension=3,
            n_grid=130,
            optimizer=OptimizerSettings(maxiter=300, coefficient_bound_factor=1.0),
            mode_selection=ModeSelectionSettings(
                modes=(1, 2, 3, 4, 5),
                patience=3,
                relative_tolerance=1.0e-2,
                warm_best=True,
                random_starts=3,
                random_scale=0.25,
                seed=314159,
            ),
        ),
        potential_gradient=gradient,
        metadata={"case": "established_curved_valley"},
    )
    save_fourier_result(fourier, output / "fourier_result.npz")
    settings = CosmoTransitionsSettings(dimension=3, maxiter=40)
    straight = run_cosmotransitions(
        potential,
        gradient,
        FALSE_VACUUM,
        TRUE_VACUUM,
        initialization="straight",
        n_path_points=80,
        settings=settings,
    )
    preconditioned = run_cosmotransitions(
        potential,
        gradient,
        FALSE_VACUUM,
        TRUE_VACUUM,
        fourier_result=fourier,
        initialization="fourier",
        n_path_points=80,
        settings=settings,
    )
    np.savez_compressed(
        output / "paths.npz",
        straight_initial_false_to_true=straight.initial_path_false_to_true,
        straight_final=straight.final_path,
        fourier_initial_false_to_true=preconditioned.initial_path_false_to_true,
        fourier_final=preconditioned.final_path,
    )
    (output / "straight.log").write_text(straight.solver_log, encoding="ascii")
    (output / "fourier.log").write_text(preconditioned.solver_log, encoding="ascii")
    payload = {
        "case": "established_curved_valley",
        "dimension": 3,
        "fourier": fourier.summary(),
        "straight": straight.summary(),
        "fourier_initialized": preconditioned.summary(),
        "relative_action_difference": abs(preconditioned.action - straight.action)
        / abs(straight.action),
    }
    (output / "integration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
