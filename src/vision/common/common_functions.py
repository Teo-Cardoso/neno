import numpy as np


def compute_angle_to_camera(
    image_x: int, camera_parameter_cx: int, focal_length: int
) -> float:
    return np.arctan2(focal_length, (image_x - camera_parameter_cx))
