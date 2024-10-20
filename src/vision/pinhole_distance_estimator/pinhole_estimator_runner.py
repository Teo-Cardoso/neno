from dataclasses import dataclass
from typing import List, Tuple
from interface.task import Runner
import common as VisioCommon
from interface.interface_types import (
    CircleCoordinatesOnImage,
    SphericalCoordinatesOnSensorFrame,
)
import numpy as np

import algos


@dataclass
class PinHoleEstimatorParameters:
    sensor_id: int
    full_frame_shape: Tuple[int, int] = (480, 640)
    focal_distance: int = 1  # mm
    camera_cx: int = 0
    camera_cy: int = 0
    ball_radius: int = 1  # mm
    min_covariance: float = 0.1


class PinHoleEstimator(Runner):
    def __init__(
        self,
        parameters: PinHoleEstimatorParameters = PinHoleEstimatorParameters(),
    ):
        super().__init__(
            input_type=List[CircleCoordinatesOnImage],
            output_types=List[Tuple[SphericalCoordinatesOnSensorFrame, np.ndarray]],
        )
        self.parameter = parameters

    def execute(
        self, input_list: List[CircleCoordinatesOnImage]
    ) -> List[Tuple[SphericalCoordinatesOnSensorFrame, np.ndarray]]:
        return [
            self.__create_polar_coordinates_from_circle(circle_coordiantes)
            for circle_coordiantes in input_list
        ]

    def __create_polar_coordinates_from_circle(
        self, circle: CircleCoordinatesOnImage
    ) -> SphericalCoordinatesOnSensorFrame:
        return (
            SphericalCoordinatesOnSensorFrame(
                distance=algos.compute_distance_to_camera(
                    self.parameter.focal_distance,
                    self.parameter.ball_radius,
                    circle.radius,
                ),
                angle=VisioCommon.compute_angle_to_camera(
                    circle.x,
                    self.parameter.camera_cx,
                    self.parameter.focal_distance,
                ),
                elevation=VisioCommon.compute_angle_to_camera(
                    circle.y,
                    self.parameter.camera_cy,
                    self.parameter.focal_distance,
                ),
                sensor_id=self.parameter.sensor_id,
            ),
            algos.compute_covariance_from_circularity(
                self.parameter.min_covariance, circle.circularity
            ),
        )
