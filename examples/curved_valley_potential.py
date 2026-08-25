"""Established two-field curved-valley case, exposed without benchmark imports."""

import numpy as np


FALSE_VACUUM = np.array([-0.6059055312950799, 0.8830215708947969])
TRUE_VACUUM = np.array([0.8942723612839004, 0.7212254614935721])


def potential(phi):
    h, s = phi[0], phi[1]
    return (
        0.25 * (h * h - 1.0) ** 2
        + 0.25 * (s * s - 1.0) ** 2
        + 0.3 * h * h * s * s
        - 0.1 * h
    )


def gradient(phi):
    point = np.asarray(phi, dtype=float)
    h, s = point[0], point[1]
    return np.array(
        [
            h * (h * h - 1.0) + 0.6 * h * s * s - 0.1,
            s * (s * s - 1.0) + 0.6 * s * h * h,
        ]
    )
