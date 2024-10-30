import time
import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple
from filterpy.kalman import KalmanFilter
from cv2.typing import MatLike

from interface.task import Runner
from interface.interface_types import (
    SphericalCoordinatesOnSensorFrame,
    BallStatus,
    BallMeasurement,
)


@dataclass
class BallStatusFilterParameters:
    sensor_id: int = 0


@dataclass
class KalmanFilterStatus:
    trustness: float = 0.0
    update_in_the_last_cycle: bool = False
    cycles_without_update: int = 0
    last_update_timestamp: int = 0
    cycles: int = 1

    def update_trustness(self, trustness_coeficient: float):
        trustness_rate = trustness_coeficient
        self.trustness = max(min(self.trustness + trustness_rate, 1.0), 0.0)


class BallStatusFilter(Runner):
    def __init__(
        self,
        parameters: BallStatusFilterParameters = BallStatusFilterParameters(),
    ):
        super().__init__(
            input_type=List[Tuple[SphericalCoordinatesOnSensorFrame, np.ndarray]],
            output_types=List[BallStatus],
        )
        self.parameter: BallStatusFilterParameters = parameters
        self.ball_status: List[BallStatus] = []
        self.kalman_filters: List[KalmanFilter] = []
        self.kalman_filters_status: List[KalmanFilterStatus] = []
        self.last_timestamp: int = 0
        self.initialized: bool = False

    # TODO Implement the execute method
    def execute(
        self,
        input_candidates: List[Tuple[int, List[BallMeasurement]]],
    ) -> BallStatus:
        if not self.initialized:
            self.initialize_filter(input_candidates)
        else:
            for sensor_input in input_candidates:
                self.run_filter_for_sensor(sensor_input)

        # Last Predict with current timestamp

        # Update BallStatus objects
        for index, kalman_filter in enumerate(self.kalman_filters):
            if index >= len(self.ball_status):
                self.ball_status.append(
                    BallStatus(id=time.time_ns(), timestamp=self.last_timestamp)
                )

            self.ball_status[index].position = self.kalman_filters[index].x
            self.ball_status[index].covariance_matrix = self.kalman_filters[index].P
            self.ball_status[index].timestamp = self.last_timestamp

        return self.ball_status

    def run_filter_for_sensor(
        self, ball_measurements: List[BallMeasurement]
    ) -> List[BallStatus]:
        # 0. Reset the status of the filters
        # for kalman_filter_status in self.kalman_filters_status:
        #    kalman_filter_status.update_in_the_last_cycle = False
        #    kalman_filter_status.cycles_without_update += 1

        if len(ball_measurements) == 0:
            return

        # 1. Update ball status predictions
        self.update_kalman_predictions(ball_measurements[0].timestamp)

        # 2. For Each measurement
        for measurement in ball_measurements:
            # 2.1 Make the associations between measurement and tracked balls
            (
                association_result,
                association_weight,
                association_index,
            ) = self.make_association_with_ball_status(
                measurement.position, measurement.covariance_matrix
            )

            # 2.2 Apply the measurement into the kalman filter if associated
            if association_result:
                self.apply_measurement_to_kalman_filter(
                    association_index=association_index,
                    association_weight=association_weight,
                    position=measurement.position,
                    covariance=measurement.covariance_matrix,
                    timestamp=measurement.timestamp,
                )
            # 2.3 Create a new tracking object if not associated
            else:
                self.create_new_tracking_object(
                    position=measurement.position,
                    covariance=measurement.covariance_matrix,
                    timestamp=measurement.timestamp,
                )

    def update_kalman_predictions(self, timestamp_ns: int) -> None:
        # Using the same model for every tracked object, maybe in the future it has a specific model
        # for each tracker so we can use other information to have a better prediction
        # Such as to be close to a robot ou hit the floor.
        # We also need to add a custom matrix Q to update the covariance using the delta time

        matrix_f = np.eye(6)
        matrix_q_base = np.eye(6)

        for index, kalman_filter in enumerate(self.kalman_filters):
            delta_time_ns = (
                timestamp_ns - self.kalman_filters_status[index].last_update_timestamp
            )
            self.update_matrix_f(matrix_f, delta_time_ns)
            matrix_q = matrix_q_base * (500 * delta_time_ns / 1e9)
            kalman_filter.predict(F=matrix_f, Q=matrix_q)
            self.kalman_filters_status[index].last_update_timestamp = timestamp_ns

    def initialize_filter(
        self, sensors_inputs: List[Tuple[int, List[BallMeasurement]]]
    ) -> None:
        self.initialized = True
        for _, sensor_measurements in sensors_inputs:
            for measurement in sensor_measurements:
                self.create_new_tracking_object(
                    measurement.position,
                    measurement.covariance_matrix,
                    measurement.timestamp,
                    trustness=0.15,
                )

    def make_associations_with_ball_status(
        self, measurements: Tuple[np.ndarray, np.ndarray]
    ) -> List[Associations]:
        associations: List[Associations] = []
        for meas_index, measurement in enumerate(measurements):
            res, _, kalman_index = self.make_association_with_ball_status(measurement)
            if res:
                associations.append(Associations(meas_index, kalman_index))

        return associations

    def make_association_with_ball_status(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[bool, float, int]:
        # Perform the association to found out each object is be measurand by the measurement.
        # Now it is done only a simple distance check, but this can be improved in the future.
        # Add new features, as color, size, velocity and others.
        # We are not using the covariance now, only the mean

        association_result: Tuple[bool, float, int] = (False, 0.0, 0)
        for kalman_filter_index, kalman_filter in enumerate(self.kalman_filters):
            # Add simple distance calculation
            distance_diff = np.linalg.norm(mean[0][:3] - kalman_filter.x[:3])
            if distance_diff > 500:
                print(f"BALL FILTER L174: Distance diff is too large: {distance_diff}")
                continue

            if not association_result[0] or association_result[1] > distance_diff:
                # Update the association result
                association_result = (True, distance_diff, kalman_filter_index)

        return association_result

    def apply_measurement_to_kalman_filter(
        self,
        association_index: int,
        association_weight: float,
        position: np.ndarray,
        covariance: np.ndarray,
        timestamp: int,
    ) -> None:
        self.kalman_filters[association_index].update(z=position, R=covariance)

        trustness_coeficient = association_weight  # This need to be better planned
        self.kalman_filters_status[association_index].update_trustness(
            trustness_coeficient
        )
        self.kalman_filters_status[association_index].update_in_the_last_cycle = True
        self.kalman_filters_status[association_index].cycles_without_update = 0
        self.kalman_filters_status[association_index].last_update_timestamp = timestamp

    def create_new_tracking_object(
        self,
        position: np.ndarray,
        covariance: np.ndarray,
        timestamp: int,
        trustness: int | None = None,
    ) -> None:
        # 1.0 Create Kalman Filter object
        self.kalman_filters.append(KalmanFilter(dim_x=6, dim_z=3))

        # 1.1 Create first measurements
        # 1.1.1 Add first position measurement
        self.kalman_filters[-1].x = np.array(
            [
                position[0],
                position[1],
                position[2],
                0.0,
                0.0,
                0.0,
            ]
        )
        # 1.1.1 Add first covariances
        self.kalman_filters[-1].P = np.block(
            [
                [covariance, np.zeros((3, 3))],
                [np.zeros((3, 3)), 500 * np.eye(3)],
            ]
        )

        # 1.2 Add H Model
        self.kalman_filters[-1].H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )

        # 1.3 Add Matrix Q ( process noise )
        # TODO This should be dynamically changed with dt
        self.kalman_filters[-1].Q = 500 * np.eye(6)

        # 2.1 Create Kalman Filter Status

        if trustness is None:
            trustness = self.compute_trustness_from_covariance(covariance=covariance)

        self.kalman_filters_status.append(
            KalmanFilterStatus(
                trustness=trustness,
                last_update_timestamp=timestamp,
                update_in_the_last_cycle=True,
            )
        )

    def compute_trustness_from_covariance(self, covariance: np.ndarray) -> float:
        MAX_TRUSTNESS = 0.5
        covariance_norm_diag = np.linalg.norm(np.diag(covariance))

        if covariance_norm_diag < 0.01:  # Zero
            return MAX_TRUSTNESS

        return MAX_TRUSTNESS * min((174 / covariance_norm_diag), 1.0)

    def update_matrix_f(self, matrix_f: np.ndarray, dt_ns: float):
        dt_s = dt_ns / 1e9
        matrix_f[0, 3] = dt_s
        matrix_f[1, 4] = dt_s
        matrix_f[2, 5] = dt_s
