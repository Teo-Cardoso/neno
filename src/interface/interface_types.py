from collections import namedtuple
from dataclasses import dataclass
from typing import List

import numpy as np


# CircleCoordinatesOnImage Definitions
CircleCoordinatesOnImage = namedtuple(
    "CircleCoordinatesOnImage", ["x", "y", "radius", "circularity"]
)
CircleCoordinatesOnImage.__new__.__defaults__ = (0, 0, 0, 0.0)


def CircleCoordinatesOnImageStr(self):
    return f"CircleCoordinatesOnImage(x={self.x}, y={self.y}, radius={self.radius}, circularity={self.circularity})"


CircleCoordinatesOnImage.__str__ = CircleCoordinatesOnImageStr


# PolarCoordinatesOnSensorFrame Definitions
PolarCoordinatesOnSensorFrame = namedtuple(
    "PolarCoordinatesOnSensorFrame", ["distance", "angle", "sensor_id"]
)
PolarCoordinatesOnSensorFrame.__new__.__defaults__ = (0, 0)


def PolarCoordinatesOnSensorFrameStr(self):
    return f"PolarCoordinatesOnSensorFrame(distance={self.distance}, angle={self.angle}, sensor_id={self.sensor_id})"


PolarCoordinatesOnSensorFrame.__str__ = PolarCoordinatesOnSensorFrameStr


# SphericalCoordinatesOnSensorFrame Definitions
SphericalCoordinatesOnSensorFrame = namedtuple(
    "SphericalCoordinatesOnSensorFrame", ["distance", "angle", "elevation", "sensor_id"]
)
SphericalCoordinatesOnSensorFrame.__new__.__defaults__ = (0, 0, 0, 0)


def SphericalCoordinatesOnSensorFrameStr(self):
    return f"SphericalCoordinatesOnSensorFrame(distance={self.distance}, angle={self.angle}, elevation={self.elevation}, sensor_id={self.sensor_id})"


SphericalCoordinatesOnSensorFrame.__str__ = SphericalCoordinatesOnSensorFrameStr


# BallMeasurement Definitions
@dataclass
class BallMeasurement:
    sensor_id: int
    timestamp: int
    position: np.ndarray
    covariance_matrix: np.ndarray

    def __str__(self):
        return f"BallMeasurement(sensor_id={self.sensor_id},  timestamp={self.timestamp}), position={self.position}, covariance_matrix={self.covariance_matrix}"


# BallStatus Definitions
@dataclass
class BallStatus:
    id: int
    timestamp: int
    position: np.ndarray = None
    coovariance_matrix: np.ndarray = None
    sensors: List[int] = None
    probability: float = 0.0

    def __str__(self):
        return f"BallStatus(id={self.id}, position={self.position}, coovariance_matrix={self.coovariance_matrix}, timestamp={self.timestamp}, sensors={self.sensors}, probability={self.probability})"
