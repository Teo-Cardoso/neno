from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import cv2
from cv2.typing import MatLike
from interface.task import Runner
from interface.interface_types import CircleCoordinatesOnImage
import numpy as np

from algos import cvt_color_and_inrange, circle_detection_hough_circles


@dataclass
class BallDetectionColorBasedParameters:
    full_frame_shape: Tuple[int, int] = (480, 640)
    lower_hsv: Tuple[int, int, int] = (0, 0, 0)
    upper_hsv: Tuple[int, int, int] = (255, 255, 255)

    run_hough_in_place: bool = False
    convert_type: int = cv2.COLOR_RGB2HSV
    ball_detection_method: Callable = circle_detection_hough_circles
    ball_detection_method_parameters: Dict[str, any] = field(
        default_factory=lambda: {
            "kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            "dp": 2.0,
            "min_dist": 100.0,
            "param1": 200,
            "param2": 0,
            "min_radius": 10,
            "max_radius": 40,
        }
    )


class BallDetectionColorBased(Runner):
    def __init__(
        self,
        parameters: BallDetectionColorBasedParameters = BallDetectionColorBasedParameters(),
    ):
        super().__init__(
            input_type=MatLike, output_types=List[CircleCoordinatesOnImage]
        )
        self.parameter = parameters

        # Pre-allocate filtered_frame
        self.filtered_frame = np.empty(
            (*self.parameter.full_frame_shape, 1), dtype=np.uint8
        )

    def execute(self, input: MatLike) -> List[CircleCoordinatesOnImage]:
        cvt_color_and_inrange(
            input,
            self.parameter.lower_hsv,
            self.parameter.upper_hsv,
            output_img=self.filtered_frame,
            convert_type=self.parameter.convert_type,
            use_input_in_place=False,
        )

        circles_result: List[
            CircleCoordinatesOnImage
        ] = self.parameter.ball_detection_method(
            self.filtered_frame, **self.parameter.ball_detection_method_parameters
        )

        if circles_result is None:
            circles_result = []

        return circles_result
