from dataclasses import dataclass
from typing import List, Tuple
from interface.task import Runner
from interface.interface_types import (
    CircleCoordinatesOnImage,
    SphericalCoordinatesOnSensorFrame,
)
import numpy as np
from cv2.typing import MatLike

import common as VisioCommon
import algos


@dataclass
class KinectEstimatorParameters:
    sensor_id: int
    focal_distance: int = 1  # mm
    camera_cx: int = 0
    camera_cy: int = 0
    min_covariance: float = 0.1


class KinectEstimator(Runner):
    def __init__(
        self,
        parameters: KinectEstimatorParameters = KinectEstimatorParameters(),
    ):
        super().__init__(
            input_type=Tuple[MatLike, List[CircleCoordinatesOnImage]],
            output_types=List[Tuple[SphericalCoordinatesOnSensorFrame, np.ndarray]],
        )
        self.parameter = parameters

    def execute(
        self, inputs: Tuple[MatLike, List[CircleCoordinatesOnImage]]
    ) -> List[Tuple[SphericalCoordinatesOnSensorFrame, np.ndarray]]:
        DEPTH_MAP_INDEX = 0
        CIRCLE_LIST_INDEX = 1

        return [
            self.__create_polar_coordinates_from_circle(
                inputs[DEPTH_MAP_INDEX], circle_coordiantes
            )
            for circle_coordiantes in inputs[CIRCLE_LIST_INDEX]
        ]

    def __create_polar_coordinates_from_circle(
        self, depth_map: MatLike, circle: CircleCoordinatesOnImage
    ) -> SphericalCoordinatesOnSensorFrame:
        circle_p = depth_map[circle.y, circle.x]
        return (
            SphericalCoordinatesOnSensorFrame(
                distance=circle_p,
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
            algos.compute_covariance_from_distance_and_circularity(
                self.parameter.min_covariance, circle_p, circle.circularity
            ),
        )
