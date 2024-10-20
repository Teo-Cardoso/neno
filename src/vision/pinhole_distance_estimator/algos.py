from numpy import diag


def compute_distance_to_camera(
    focal_length: int, real_object_width: int, image_object_width: int
) -> int:
    if image_object_width == 0:
        return float("inf")

    return (real_object_width * focal_length) / image_object_width


def compute_covariance_from_circularity(
    min_covariance: float, circularity: float
) -> float:
    if circularity < 0.01:
        return diag([float("inf"), float("inf"), float("inf")])

    covariance_scalar = (
        min_covariance + (1.0 - circularity) / circularity
    )  # TODO: Adjust this formula
    return diag([covariance_scalar, covariance_scalar, covariance_scalar])
