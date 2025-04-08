import numpy as np


def circle_from_points(p1, p2, p3):
    # Convert points to numpy arrays
    A = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1]
    ])

    B = np.array([
        -(p1[0] ** 2 + p1[1] ** 2),
        -(p2[0] ** 2 + p2[1] ** 2),
        -(p3[0] ** 2 + p3[1] ** 2)
    ])

    # Solve using least squares (to be safe in degenerate cases)
    X = np.linalg.lstsq(A, B, rcond=None)[0]

    # Circle center coordinates
    cx = -0.5 * X[0]
    cy = -0.5 * X[1]

    return cx, cy
