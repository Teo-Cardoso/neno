import numpy as np


def compute_covariance_from_distance_and_circularity(
    min_covariance: float, distance: float, circularity: float
) -> float:
    if circularity < 0.01:
        return np.diag([float("inf"), float("inf"), float("inf")])

    covariance_scalar = (
        min_covariance + distance / 100.0 + (1.0 - circularity) / circularity
    )  # TODO: Adjust this formula
    return np.diag([covariance_scalar, covariance_scalar, covariance_scalar])
