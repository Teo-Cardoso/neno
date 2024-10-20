import time
from typing import List

import numpy as np
from vision.ball_detection_color_based.ball_detection_runner import (
    BallDetectionColorBased,
    BallDetectionColorBasedParameters,
)
import cv2

import algos
from interface.interface_types import CircleCoordinatesOnImage
import freenect

# from scalene.scalene_profiler import enable_profiling


def run_cvtColorInRange_example():
    input_image = cv2.imread("/workspaces/ws/neno_ws/src/res/imgs/test_image_77.png")
    output = algos.cvt_color_and_inrange(
        input_img=input_image,
        lower_hsv=(4, 0, 0),
        upper_hsv=(31, 214, 250),
        use_input_in_place=False,
    )

    if output is not None:
        cv2.imshow("run_cvtColorInRange_example_input", input_image)
        cv2.imshow("run_cvtColorInRange_example_output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_circle_detection_contours_example():
    print("Running run_circle_detection_contours_example")
    input_image = cv2.imread("/workspaces/ws/neno_ws/src/res/imgs/test_image_77.png")
    gray_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    filtered_frame = cv2.equalizeHist(gray_input)
    cv2.imshow("input", gray_input)
    cv2.imshow("filtered_frame", filtered_frame)
    cv2.waitKey(0)
    input_cvt_in_range = algos.cvt_color_and_inrange(
        input_img=input_image,
        lower_hsv=(4, 0, 0),
        upper_hsv=(31, 214, 250),
        use_input_in_place=False,
    )

    output: List[CircleCoordinatesOnImage] = algos.circle_detection_contours(
        input_img=input_cvt_in_range,
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
        min_radius=20,
        max_radius=50,
    )

    if output is None:
        return

    for circle in output:
        cv2.circle(
            input_image,
            (int(circle.x), int(circle.y)),
            int(circle.radius),
            (0, 255, 0),
            2,
        )
        cv2.circle(input_image, (int(circle.x), int(circle.y)), 2, (0, 0, 255), 3)

    cv2.imshow("run_circle_detection_contours_example", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_circle_detection_hough_circles_example():
    print("Running run_circle_detection_hough_circles_example")
    input_image = cv2.imread("/workspaces/ws/neno_ws/src/res/imgs/test_image_77.png")
    input_cvt_in_range = algos.cvt_color_and_inrange(
        input_img=input_image,
        lower_hsv=(4, 0, 0),
        upper_hsv=(31, 214, 250),
        use_input_in_place=False,
    )

    output: List[CircleCoordinatesOnImage] = algos.circle_detection_hough_circles(
        input=input_cvt_in_range,
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
    )
    if output is None:
        return

    for circle in output:
        cv2.circle(
            input_image,
            (int(circle.x), int(circle.y)),
            int(circle.radius),
            (0, 255, 0),
            2,
        )
        cv2.circle(input_image, (int(circle.x), int(circle.y)), 2, (0, 0, 255), 3)
    cv2.imshow("run_circle_detection_hough_circles_example", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_example():
    parameters = BallDetectionColorBasedParameters()
    parameters.lower_hsv = (4, 114, 80)
    parameters.upper_hsv = (31, 214, 250)
    parameters.convert_type = cv2.COLOR_RGB2HSV
    parameters.ball_detection_method_parameters = {
        "kernel": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        "max_radius": 100,
    }
    ball_detection_color_based = BallDetectionColorBased(parameters)

    using_kinect = True
    cap = None
    if not using_kinect:
        cap = cv2.VideoCapture(0)  # MatLike source

    counter = 0
    ticker_count = 0
    while counter < 2000:
        counter += 1

        ret, frame = (False, None)
        if using_kinect:
            ret, frame = (True, freenect.sync_get_video()[0])
            pass
        else:
            ret, frame = cap.read()

        if not ret:
            break

        # with enable_profiling():
        time_before = cv2.getTickCount()
        detected_balls = ball_detection_color_based.execute(frame)
        ticker_count += cv2.getTickCount() - time_before

        for ball in detected_balls:
            cv2.circle(
                frame, (int(ball.x), int(ball.y)), int(ball.radius), (0, 255, 0), 2
            )
            cv2.circle(frame, (int(ball.x), int(ball.y)), 2, (0, 0, 255), 3)

        cv2.imshow("Frame Origin", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(
        f"Time: {((ticker_count) / cv2.getTickFrequency()) / (counter + 1)}s per frame"
    )

    if not using_kinect:
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_cvtColorInRange_example()
    run_circle_detection_hough_circles_example()
    run_circle_detection_contours_example()
    run_example()
