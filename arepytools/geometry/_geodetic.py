# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geodetic module
---------------
"""

import numpy as np
import numpy.typing as npt

from arepytools.geometry.conversions import llh2xyz, xyz2llh
from arepytools.geometry.ellipsoid import WGS84

_semi_axis_ratio_sqr = WGS84.semi_axes_ratio_min_max**2
_major_semi_axis_sqr = WGS84.semi_major_axis**2
_minor_semi_axis_sqr = WGS84.semi_minor_axis**2


def _ze(x: float, y: float) -> float:
    eps = 1.0e-21

    return (
        np.sqrt(_minor_semi_axis_sqr - _semi_axis_ratio_sqr * (x**2 + y**2)) + eps
    )


def _ze_x(x: float, y: float) -> float:
    return -WGS84.semi_axes_ratio_min_max * x / _ze(x, y)


def _ze_xy(x: float, y: float) -> float:
    return (_semi_axis_ratio_sqr * x) / _ze(x, y) ** 2 * _ze_x(y, x)


def _ze_xx(x: float, y: float) -> float:
    return -_semi_axis_ratio_sqr / _ze(x, y) + (_semi_axis_ratio_sqr * x) / _ze(
        x, y
    ) ** 2 * _ze_x(x, y)


def _compute_geodetic_jacobian(
    point: np.ndarray, sensor_position: np.ndarray
) -> np.ndarray:
    jac = np.empty(shape=(3, 3), dtype=float)

    zed_diff = point[2] - sensor_position[2]

    jac[0][0] = 2.0 * point[0] / _major_semi_axis_sqr
    jac[0][1] = 2.0 * point[1] / _major_semi_axis_sqr
    jac[0][2] = 2.0 * point[2] / _minor_semi_axis_sqr
    jac[1][0] = 1.0 + zed_diff * _ze_xx(point[0], point[1])
    jac[1][1] = zed_diff * _ze_xy(point[0], point[1])
    jac[1][2] = _ze_x(point[0], point[1])
    jac[2][0] = jac[1][1]
    jac[2][1] = 1.0 + zed_diff * _ze_xx(point[1], point[0])
    jac[2][2] = _ze_x(point[1], point[0])

    return jac


def _compute_geodetic_rhs(
    point: npt.NDArray[np.floating], sensor_position: npt.NDArray[np.floating]
) -> np.ndarray:
    rhs = np.empty(shape=(3,), dtype=float)
    los = point - sensor_position
    rhs[0] = (
        (point[0] ** 2 + point[1] ** 2) / _major_semi_axis_sqr
        + point[2] ** 2 / _minor_semi_axis_sqr
        - 1
    )
    rhs[1] = los[0] + los[2] * _ze_x(point[0], point[1])
    rhs[2] = los[1] + los[2] * _ze_x(point[1], point[0])
    return rhs


def compute_geodetic_point(sensor_positions: npt.ArrayLike) -> np.ndarray:
    """Compute the geodetic point that corresponds to a sensor position

    :param sensor_position: sensor position array like with shape (3,) or (N, 3)

    :return: geodetic point as a (3,) or (N, 3) numpy array
    """
    sensor_positions = np.asarray(sensor_positions)

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(
            f"sensor_positions has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)"
        )

    sensor_positions = sensor_positions.copy()
    change_sign = sensor_positions[..., 2] < 0
    sensor_positions[change_sign, 2] *= -1

    geodetic_points = xyz2llh(sensor_positions.T)
    geodetic_points[2, ...] = 0
    geodetic_points = llh2xyz(geodetic_points).T.reshape(sensor_positions.shape)

    for sensor_position, point in zip(
        sensor_positions.reshape((-1, 3)),
        geodetic_points.reshape((-1, 3)),
    ):
        increment_norm_threshold = 1e-9
        max_num_iterations = 10
        for _ in range(max_num_iterations):
            jacobian = _compute_geodetic_jacobian(point[:], sensor_position)
            rhs = _compute_geodetic_rhs(point[:], sensor_position)

            increment = -np.linalg.solve(jacobian, rhs)

            increment_norm = np.linalg.norm(increment)

            point[:] += increment

            if increment_norm < increment_norm_threshold:
                break

    geodetic_points[change_sign, 2] *= -1

    return geodetic_points
