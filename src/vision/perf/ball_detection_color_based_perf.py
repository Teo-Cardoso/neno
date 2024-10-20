from ball_detection_color_based import (
    BallDetectionColorBased,
    algos,
    BallDetectionColorBasedParameters,
)
import cv2

import pytest


@pytest.mark.parametrize(
    "n, hsv_lower, hsv_upper",
    [
        (0, (4, 145, 81), (31, 214, 250)),
        (33, (5, 150, 85), (32, 220, 255)),
        (66, (6, 155, 90), (33, 225, 260)),
        (99, (7, 160, 95), (34, 230, 265)),
    ],
)
def test_cvt_color_and_inrange(benchmark, n, hsv_lower, hsv_upper):
    benchmark(
        algos.cvt_color_and_inrange,
        cv2.imread(f"/workspaces/ws/neno_ws/src/res/imgs/test_image_{n}.png").copy(),
        hsv_lower,
        hsv_upper,
        cv2.COLOR_RGB2HSV,
        False,
    )


@pytest.mark.parametrize("n", [0, 34, 67, 98])
def test_circle_detection_hough_circles(benchmark, n):
    benchmark(
        algos.circle_detection_hough_circles,
        cv2.imread(
            f"/workspaces/ws/neno_ws/src/res/imgs/test_image_{n}.png",
            cv2.IMREAD_GRAYSCALE,
        ),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )


@pytest.mark.parametrize("n", [20, 40, 60, 80, 97])
def test_BallDetectionColorBased(benchmark, n):
    parameters = BallDetectionColorBasedParameters()
    parameters.lower_hsv = (4, 145, 81)
    parameters.upper_hsv = (31, 214, 250)
    parameters.convert_type = cv2.COLOR_RGB2HSV
    ball_detection_color_based = BallDetectionColorBased(parameters)

    benchmark(
        ball_detection_color_based.execute,
        cv2.imread(
            f"/workspaces/ws/neno_ws/src/res/imgs/test_image_{n}.png",
            cv2.IMREAD_COLOR,
        ),
    )
