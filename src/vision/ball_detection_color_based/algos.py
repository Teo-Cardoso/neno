from collections import namedtuple
import cv2
from typing import List, Tuple
import numpy as np
from math import pi as MATH_PI
from interface.interface_types import CircleCoordinatesOnImage


def cvt_color_and_inrange(
    input_img: cv2.typing.MatLike,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int],
    convert_type: int = cv2.COLOR_RGB2HSV,
    output_img: cv2.typing.MatLike = None,
    use_input_in_place: bool = False,
) -> cv2.typing.MatLike | None:
    """
    Converts an image from {convert_type} to HSV and then applies an inRange filter to it.
    """

    using_output_img: bool = output_img is not None
    if not using_output_img:
        output_img = np.empty(
            (input_img.shape[0], input_img.shape[1], 1), dtype=np.uint8
        )

    input_tmp_link = input_img
    if use_input_in_place:
        cv2.cvtColor(input_img, convert_type, input_tmp_link)
    else:
        input_tmp_link = cv2.cvtColor(input_img, convert_type)

    cv2.inRange(input_tmp_link, lower_hsv, upper_hsv, output_img)

    if not using_output_img:
        return output_img


def circle_detection_hough_circles(
    input: cv2.typing.MatLike,
    kernel: cv2.typing.MatLike,
    dp: float = 2,  # TODO: Add possibility to dynamicly change dp based on image size
    min_dist: float = 100,
    param1: int = 200,
    param2: int = 0.5,
    min_radius: int = 10,
    max_radius: int = 40,
) -> np.ndarray | None:
    """
    Detects circles in an image using the HoughCircles method.
    First it applies a morphological close operation to the input image, then it applies a GaussianBlur to the result
    """
    if input is None or input.size == 0:
        return None

    filtered_frame = cv2.morphologyEx(input, cv2.MORPH_CLOSE, kernel)

    # TODO: Decide which filter to use
    cv2.GaussianBlur(filtered_frame, (9, 9), 0, filtered_frame)
    cv2.imshow("filtered_frame", filtered_frame)
    # filtered_frame = cv2.bilateralFilter(filtered_frame, 5, 75, 75)

    hough_circles = cv2.HoughCircles(
        filtered_frame,
        cv2.HOUGH_GRADIENT_ALT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if hough_circles is None:
        return []

    return [
        CircleCoordinatesOnImage(
            x=circle[0][0], y=circle[0][1], radius=circle[0][2], circularity=1.0
        )
        for circle in hough_circles
    ]


def circle_detection_contours(
    input_img: cv2.typing.MatLike,
    kernel: cv2.typing.MatLike,
    min_radius: int = 10,
    max_radius: int = 40,
) -> List[CircleCoordinatesOnImage]:
    """
    Detects circles in an image using the contours method.
    First it applies a morphological close operation to the input image, then it applies a GaussianBlur to the result
    """
    if input_img is None or input_img.size == 0:
        return []

    filtered_frame = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, kernel)
    cv2.GaussianBlur(filtered_frame, (9, 9), 0, filtered_frame)
    cv2.Canny(filtered_frame, 50, 150, filtered_frame)

    # Find contours
    contours, _ = cv2.findContours(
        filtered_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    circles: List[CircleCoordinatesOnImage] = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not (
            radius > min_radius and radius < max_radius
        ):  # Filter small and big circles
            continue

        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * MATH_PI * contour_area / perimeter**2
        if abs(1 - circularity) > 0.2:  # Filter non-circular shapes
            continue

        circles.append(
            CircleCoordinatesOnImage(x=x, y=y, radius=radius, circularity=circularity)
        )

    return circles
