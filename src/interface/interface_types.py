from collections import namedtuple


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
