"""A small curved two-field potential for public API examples."""

import numpy as np


FALSE_VACUUM = np.array([-0.9456492739235919, -0.03701160775472421])
TRUE_VACUUM = np.array([1.0466805318046022, 0.033439047480567696])


def potential(phi):
    x, y = phi[0], phi[1]
    valley = y - 0.35 * (x * x - 1.0)
    return 0.25 * (x * x - 1.0) ** 2 + 0.5 * valley**2 - 0.1 * x


def gradient(phi):
    point = np.asarray(phi, dtype=float)
    x, y = point[0], point[1]
    valley = y - 0.35 * (x * x - 1.0)
    return np.array([x * (x * x - 1.0) - 0.7 * x * valley - 0.1, valley])
