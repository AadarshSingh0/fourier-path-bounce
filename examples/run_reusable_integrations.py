"""Run auditable straight/Fourier CosmoTransitions and FindBounce exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.reusable_potential import FALSE_VACUUM, TRUE_VACUUM, gradient, potential
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
            dimension=4,
            n_grid=130,
            optimizer=OptimizerSettings(maxiter=300),
            mode_selection=ModeSelectionSettings(modes=(1, 2, 3), patience=2),
        ),
        potential_gradient=gradient,
        metadata={"example": "reusable_two_field"},
    )
    save_fourier_result(fourier, output / "fourier_result.npz")
    ct_settings = CosmoTransitionsSettings(maxiter=40)
    straight = run_cosmotransitions(
        potential,
        gradient,
        FALSE_VACUUM,
        TRUE_VACUUM,
        initialization="straight",
        n_path_points=80,
        settings=ct_settings,
    )
    preconditioned = run_cosmotransitions(
        potential,
        gradient,
        FALSE_VACUUM,
        TRUE_VACUUM,
        fourier_result=fourier,
        initialization="fourier",
        n_path_points=80,
        settings=ct_settings,
    )
    np.savez_compressed(
        output / "cosmotransitions_paths.npz",
        straight_initial_false_to_true=straight.initial_path_false_to_true,
        straight_final=straight.final_path,
        fourier_initial_false_to_true=preconditioned.initial_path_false_to_true,
        fourier_final=preconditioned.final_path,
    )
    (output / "cosmotransitions_straight.log").write_text(straight.solver_log, encoding="ascii")
    (output / "cosmotransitions_fourier.log").write_text(preconditioned.solver_log, encoding="ascii")
    payload = {
        "potential": "examples.reusable_potential:potential",
        "gradient": "examples.reusable_potential:gradient",
        "false_vacuum": FALSE_VACUUM.tolist(),
        "true_vacuum": TRUE_VACUUM.tolist(),
        "fourier": fourier.summary(),
        "cosmotransitions_straight": straight.summary(),
        "cosmotransitions_fourier": preconditioned.summary(),
        "cosmotransitions_relative_action_difference": abs(
            preconditioned.action - straight.action
        )
        / abs(straight.action),
    }
    (output / "cosmotransitions_integration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    for k in (1, 4):
        point_set = prepare_findbounce_points(fourier, k, sampling="arclength")
        export_findbounce_points(point_set, output / f"findbounce_K{k}", basename="initializer")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
