# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Reference frames module
-----------------------
"""

from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np
import numpy.typing as npt

from arepytools.constants import SECONDS_IN_A_DAY
from arepytools.geometry._geodetic import compute_geodetic_point
from arepytools.geometry.rotation import RotationOrder, compute_rotation


class ReferenceFrame(Enum):
    """Available reference frames"""

    geocentric = "GEOCENTRIC"
    geodetic = "GEODETIC"
    zero_doppler = "ZERODOPPLER"


ReferenceFrameLike = Union[str, ReferenceFrame]
""":class:`ReferenceFrame` like type hint """

_earth_angular_velocity = 2.0 * np.pi / SECONDS_IN_A_DAY


def compute_sensor_local_axis(
    positions: npt.ArrayLike,
    velocities: npt.ArrayLike,
    reference_frame: ReferenceFrameLike,
) -> np.ndarray:
    """Compute the axis of the local reference frame

    Parameters
    ----------
    positions : npt.ArrayLike
        (3,) or (N, 3) one or more sensor position
    velocities : npt.ArrayLike
        (3,) or (N, 3) one or more sensor velocity
    reference_frame : ReferenceFrameLike
        reference frame

    Returns
    -------
    np.ndarray
         (3, 3) or (N, 3, 3) one or more sensor local axis

    Examples
    --------

    single position and velocity

    >>> print(position.shape)
    (3,)
    >>> axis = compute_sensor_local_axis(position, velocity, ReferenceFrame.zero_doppler)
    >>> print(axis.shape)
    (3, 3)

    multiple position and velocity

    >>> print(positions.shape)
    (10, 3)
    >>> axes = compute_sensor_local_axis(positions, velocities, ReferenceFrame.zero_doppler)
    >>> print(axes.shape)
    (10, 3, 3)

    reference frame as string

    >>> compute_sensor_local_axis(position, velocity, "ZERODOPPLER")
    """

    reference_frame = ReferenceFrame(reference_frame)

    if reference_frame == ReferenceFrame.zero_doppler:
        return compute_zerodoppler_reference_frame(positions, velocities)

    if reference_frame == ReferenceFrame.geocentric:
        return compute_geocentric_reference_frame(positions, velocities)

    if reference_frame == ReferenceFrame.geodetic:
        return compute_geodetic_reference_frame(positions, velocities)

    raise ValueError("Unknown reference frame")  # pragma: no cover


def compute_zerodoppler_reference_frame(
    sensor_position: npt.ArrayLike, sensor_velocity: npt.ArrayLike
) -> np.ndarray:
    """Compute the ZeroDoppler reference frame

    Reference frame
    - x-versor oriented as sensor non-inertial velocity
    - y-versor given by the cross product between x and sensor position corrected with Earth eccentricity
    - z-versor completing the reference frame

    - output frame has x as first column, y as second one and z as the last one.

    Parameters
    ----------
    sensor_position, sensor_velocity : npt.ArrayLike
        array like with shape (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        the reference frame as (3,3) or (N, 3, 3) np array

    Raises
    ------
    ValueError
        in case of invalid input
    """

    sensor_position = np.asarray(sensor_position)
    sensor_velocity = np.asarray(sensor_velocity)

    if sensor_position.shape != sensor_velocity.shape:
        raise ValueError(
            f"sensor_position and sensor_velocity have different shapes {sensor_position.shape} != {sensor_velocity.shape}"
        )

    if sensor_position.ndim > 2 or sensor_position.shape[-1] != 3:
        raise ValueError(
            f"sensor_position has invalid shape: {sensor_position.shape}, it should be (3,) or (N, 3)"
        )

    versor_x = sensor_velocity / np.linalg.norm(sensor_velocity, axis=-1, keepdims=True)

    adjusted_position = sensor_position.copy()
    beta = 0.0060611
    adjusted_position[..., 2] *= 1.0 + beta

    versor_y = np.cross(versor_x, adjusted_position)
    versor_y = versor_y / np.linalg.norm(versor_y, axis=-1, keepdims=True)

    versor_z = np.cross(versor_x, versor_y)
    versor_z = versor_z / np.linalg.norm(versor_z, axis=-1, keepdims=True)

    return np.stack([versor_x, versor_y, versor_z], axis=-1)


def compute_inertial_velocity(
    sensor_position: npt.ArrayLike, sensor_velocity: npt.ArrayLike
) -> np.ndarray:
    """Compute the sensor inertial velocity

    Parameters
    ----------
    sensor_position, sensor_velocity : npt.ArrayLike

    Returns
    -------
    np.ndarray
        the inertial velocity

    Raises
    ------
    ValueError
        in case of invalid input
    """

    sensor_position = np.asarray(sensor_position)
    sensor_velocity = np.asarray(sensor_velocity)

    intertial_velocity = sensor_velocity.copy()
    intertial_velocity[..., 0] += -_earth_angular_velocity * sensor_position[..., 1]
    intertial_velocity[..., 1] += _earth_angular_velocity * sensor_position[..., 0]

    return intertial_velocity


def compute_geocentric_reference_frame(
    sensor_position: npt.ArrayLike, sensor_velocity: npt.ArrayLike
) -> np.ndarray:
    """Computed the geocentric frame of reference

    Parameters
    ----------
    sensor_position, sensor_velocity : npt.ArrayLike
        array like with shape (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        the reference frame as (3,3) or (N, 3, 3) np array

    Raises
    ------
    ValueError
        in case of invalid input
    """
    # x-versor completing the reference frame
    # y-versor given by the cross product between z and sensor inertial velocity
    # z-versor oriented as -Psat

    sensor_position = np.asarray(sensor_position)
    sensor_velocity = np.asarray(sensor_velocity)

    if sensor_position.shape != sensor_velocity.shape:
        raise ValueError(
            f"sensor_position and sensor_velocity have different shapes {sensor_position.shape} != {sensor_velocity.shape}"
        )

    if sensor_position.ndim > 2 or sensor_position.shape[-1] != 3:
        raise ValueError(
            f"sensor_position has invalid shape: {sensor_position.shape}, it should be (3,) or (N, 3)"
        )

    versor_z = -sensor_position
    versor_z = versor_z / np.linalg.norm(versor_z, axis=-1, keepdims=True)

    intertial_velocity = compute_inertial_velocity(sensor_position, sensor_velocity)

    versor_y = np.cross(versor_z, intertial_velocity)
    versor_y = versor_y / np.linalg.norm(versor_y, axis=-1, keepdims=True)

    versor_x = np.cross(versor_y, versor_z)
    versor_x = versor_x / np.linalg.norm(versor_x, axis=-1, keepdims=True)

    return np.stack([versor_x, versor_y, versor_z], axis=-1)


def compute_geodetic_reference_frame(
    sensor_position: npt.ArrayLike, sensor_velocity: npt.ArrayLike
) -> np.ndarray:
    """Computed the geodetic frame of reference

    Parameters
    ----------
    sensor_position, sensor_velocity : np.ndarray
        array like with shape (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        the reference frame as (3,3) or (N, 3, 3) np array

    Raises
    ------
    ValueError
        in case of invalid input
    """
    sensor_position = np.asarray(sensor_position)
    sensor_velocity = np.asarray(sensor_velocity)

    if sensor_position.shape != sensor_velocity.shape:
        raise ValueError(
            f"sensor_position and sensor_velocity have different shapes {sensor_position.shape} != {sensor_velocity.shape}"
        )

    if sensor_position.ndim > 2 or sensor_position.shape[-1] != 3:
        raise ValueError(
            f"sensor_position has invalid shape: {sensor_position.shape}, it should be (3,) or (N, 3)"
        )

    geodetic_point = compute_geodetic_point(sensor_position)

    versor_z = geodetic_point - sensor_position
    versor_z = versor_z / np.linalg.norm(versor_z, axis=-1, keepdims=True)

    geocentric_frame = compute_geocentric_reference_frame(
        sensor_position, sensor_velocity
    )

    z_geocentric = np.einsum("...jk, ...j->...k", geocentric_frame, versor_z)
    z_geocentric = z_geocentric / np.linalg.norm(z_geocentric, axis=-1, keepdims=True)

    beta = -np.arctan2(z_geocentric[..., 1], z_geocentric[..., 2])

    rotation = compute_rotation(
        RotationOrder.ypr,
        yaw=np.zeros_like(beta),
        pitch=np.zeros_like(beta),
        roll=beta,
    )

    rotated_frame = np.matmul(geocentric_frame, rotation.as_matrix())

    z_rotated = np.einsum("...jk, ...j->...k", rotated_frame, versor_z)
    z_rotated = z_rotated / np.linalg.norm(z_rotated, axis=-1, keepdims=True)
    xsi = np.arctan2(z_rotated[..., 0], z_rotated[..., 2])

    second_rotation = compute_rotation(
        RotationOrder.ypr, yaw=np.zeros_like(xsi), pitch=xsi, roll=np.zeros_like(xsi)
    )

    return np.matmul(rotated_frame, second_rotation.as_matrix())


__all__ = [
    "ReferenceFrame",
    "compute_sensor_local_axis",
]
