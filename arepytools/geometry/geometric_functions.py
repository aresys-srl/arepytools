# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geometric functions module
--------------------------
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from arepytools.geometry.conversions import llh2xyz, xyz2llh
from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve
from arepytools.geometry.direct_geocoding import (
    GeocodingSide,
    direct_geocoding_monostatic,
)
from arepytools.timing.precisedatetime import PreciseDateTime


def compute_incidence_angles_from_trajectory(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    range_times: Union[float, npt.ArrayLike],
    look_direction: Union[str, GeocodingSide],
) -> Union[float, np.ndarray]:
    """Compute incidence angles from a trajectory curve compliant with the TwiceDifferentiable3DCurve protocol.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        trajectory 3D curve protocol-compliant
    azimuth_time : PreciseDateTime
        azimuth time
    range_times : Union[float, npt.ArrayLike]
        range times array like or float
    look_direction : Union[str, GeocodingSide]
        sensor look direction

    Returns
    -------
    Union[float, np.ndarray]
        incidence angles for each range time computed at the given azimuth time
    """
    look_direction = GeocodingSide(look_direction)
    sensor_pos = trajectory.evaluate(azimuth_time)
    sensor_vel = trajectory.evaluate_first_derivatives(azimuth_time)

    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_pos,
        sensor_velocities=sensor_vel,
        range_times=range_times,
        geocoding_side=look_direction.value,
        frequencies_doppler_centroid=0,
        wavelength=1,
        geodetic_altitude=0,
    )

    return compute_incidence_angles(sensor_positions=sensor_pos, points=ground_points)


def compute_look_angles_from_trajectory(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    range_times: Union[float, npt.ArrayLike],
    look_direction: Union[str, GeocodingSide],
) -> Union[float, np.ndarray]:
    """Compute look angles from a trajectory curve compliant with the TwiceDifferentiable3DCurve protocol.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        trajectory 3D curve protocol-compliant
    azimuth_time : PreciseDateTime
        azimuth time
    range_times : Union[float, npt.ArrayLike]
        range times array like or float
    look_direction : Union[str, GeocodingSide]
        sensor look direction

    Returns
    -------
    Union[float, np.ndarray]
        look angles for each range time computed at the given azimuth time
    """
    look_direction = GeocodingSide(look_direction)
    sensor_pos = trajectory.evaluate(azimuth_time)
    sensor_vel = trajectory.evaluate_first_derivatives(azimuth_time)

    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_pos,
        sensor_velocities=sensor_vel,
        range_times=range_times,
        geocoding_side=look_direction.value,
        frequencies_doppler_centroid=0,
        wavelength=1,
        geodetic_altitude=0,
    )

    sensor_position_ground = xyz2llh(sensor_pos)
    sensor_position_ground[2] = 0.0
    sensor_position_ground = llh2xyz(sensor_position_ground).squeeze()

    nadir = sensor_position_ground - sensor_pos

    return compute_look_angles(
        sensor_positions=sensor_pos, nadir_directions=nadir.T, points=ground_points
    )


def compute_look_angles(
    sensor_positions: npt.ArrayLike,
    nadir_directions: npt.ArrayLike,
    points: npt.ArrayLike,
    *,
    assume_nadir_directions_normalized: bool = False,
) -> Union[float, np.ndarray]:
    """Compute the look angles.

    Parameters
    ----------
    sensor_positions : npt.ArrayLike
        (3,) or (N, 3) one or more sensor positions
    nadir_directions : npt.ArrayLike
        (3,) or (N, 3) one or more nadir directions
    points : npt.ArrayLike
        (3,) or (N, 3) one or more points
    assume_nadir_directions_normalized : bool, optional
        True to skip nadir directions normalization, by default False

    Returns
    -------
    Union[float, np.ndarray]
        scalar or (N,) look angles in radians

    Raises
    ------
    ValueError
        in case of invalid input

    Examples
    --------

    1 position, nadir -- 1 point

    >>> look_angle = compute_look_angles(position, nadir_dir, point)

    N positions, nadirs -- 1 point -- point is broadcasted

    >>> look_angles = compute_look_angles(positions, nadir_directions, point)

    1 position, nadir -- N points -- position and nadir are broadcasted

    >>> look_angles = compute_look_angles(position, nadir_dir, points)

    N positions, nadirs -- N points

    >>> look_angles = compute_look_angles(positions, nadir_directions, points)

    Skip normalization of nadir direction

    >>> look_angle = compute_look_angles(position, nadir_dir, point, assume_nadir_directions_normalized=True)

    """
    sensor_positions = np.asarray(sensor_positions)
    nadir_directions = np.asarray(nadir_directions)
    points = np.asarray(points)

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(
            f"sensor_positions has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)"
        )

    if nadir_directions.ndim > 2 or nadir_directions.shape[-1] != 3:
        raise ValueError(
            f"nadir_directions has invalid shape: {nadir_directions.shape}, it should be (3,) or (N, 3)"
        )

    if points.ndim > 2 or points.shape[-1] != 3:
        raise ValueError(
            f"points has invalid shape: {points.shape}, it should be (3,) or (N, 3)"
        )

    los_directions = points - sensor_positions
    los_directions = los_directions / np.linalg.norm(
        los_directions, axis=-1, keepdims=True
    )

    if not assume_nadir_directions_normalized:
        nadir_directions = nadir_directions / np.linalg.norm(
            nadir_directions, axis=-1, keepdims=True
        )

    look_angle_cosinuses = np.sum(nadir_directions * los_directions, axis=-1)

    return np.arccos(np.clip(look_angle_cosinuses, a_min=-1.0, a_max=1.0))


def compute_incidence_angles(
    sensor_positions: npt.ArrayLike,
    points: npt.ArrayLike,
    *,
    surface_normals: Optional[npt.ArrayLike] = None,
    assume_surface_normals_normalized: bool = False,
) -> Union[float, np.ndarray]:
    """Compute the incidence angles

    If surface normals are not specified, points are used to define the surface normals

    .. code-block:: python

        surface_normals = points
        assume_surface_normals_normalized = False

    Parameters
    ----------
    sensor_positions : npt.ArrayLike
        (3,) or (N, 3) one or more sensor positions
    points : npt.ArrayLike
        (3,) or (N, 3) one or more points
    surface_normals : Optional[npt.ArrayLike], optional
        (3,) or (N, 3) one or more surface normal directions, by default None
    assume_surface_normals_normalized : bool, optional
        True to skip surface normals normalization, by default False

    Returns
    -------
    Union[float, np.ndarray]
        scalar or (N,) look angles in radians

    Raises
    ------
    ValueError
        in case of invalid input

    Examples
    --------

    1 position -- 1 point

    >>> incidence_angle = compute_incidence_angles(position, point)

    N positions -- 1 point -- point is broadcasted

    >>> incidence_angles = compute_incidence_angles(positions, point)

    1 position -- N points -- position is broadcasted

    >>> incidence_angles = compute_incidence_angles(position, points)

    N positions -- N points

    >>> incidence_angles = compute_incidence_angles(positions, points)

    User defined surface normal with normalization skipping

    >>> incidence_angle = compute_incidence_angles(position, point,
            surface_normals=surf_norm,
            assume_nadir_directions_normalized=True)
    """

    sensor_positions = np.asarray(sensor_positions)
    points = np.asarray(points)

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(
            f"sensor_positions has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)"
        )

    if points.ndim > 2 or points.shape[-1] != 3:
        raise ValueError(
            f"points has invalid shape: {points.shape}, it should be (3,) or (N, 3)"
        )

    if surface_normals is not None:
        surface_normals = np.asarray(surface_normals)
        if surface_normals.ndim > 2 or surface_normals.shape[-1] != 3:
            raise ValueError(
                f"surface_normals has invalid shape: {surface_normals.shape}, it should be (3,) or (N, 3)"
            )

    los_directions = points - sensor_positions
    los_directions = los_directions / np.linalg.norm(
        los_directions, axis=-1, keepdims=True
    )

    if surface_normals is None:
        surface_normals = points
        assume_surface_normals_normalized = False

    if not assume_surface_normals_normalized:
        surface_normals = surface_normals / np.linalg.norm(
            surface_normals, axis=-1, keepdims=True
        )

    incidence_angle_cosinus = -1.0 * np.sum(surface_normals * los_directions, axis=-1)

    return np.arccos(np.clip(incidence_angle_cosinus, a_min=-1.0, a_max=1.0))


def get_geometric_squint(
    sensor_positions: npt.ArrayLike,
    sensor_velocities: npt.ArrayLike,
    ground_points: npt.ArrayLike,
) -> Union[float, np.ndarray]:
    """Evaluating squint angle geometrically.

    Parameters
    ----------
    sensor_positions : npt.ArrayLike
        sensor positions array, in the form (3,) or (N, 3)
    sensor_velocities : npt.ArrayLike
        sensor velocities array, in the form (3,) or (N, 3)
    ground_points : npt.ArrayLike
        ground points array, in the form (3,) or (N, 3)

    Returns
    -------
    Union[float, np.ndarray]
        squint angle in radians
    """

    # converting inputs to arrays
    sensor_positions = np.asarray(sensor_positions)
    sensor_velocities = np.asarray(sensor_velocities)
    ground_points = np.asarray(ground_points)

    # evaluating squint angle
    line_of_sight = ground_points - sensor_positions
    line_of_sight = line_of_sight / np.linalg.norm(
        line_of_sight, axis=-1, keepdims=True
    )
    sensor_velocity_norm = sensor_velocities / np.linalg.norm(
        sensor_velocities, axis=-1, keepdims=True
    )
    squint_angle = np.arcsin(np.sum(line_of_sight * sensor_velocity_norm, axis=-1))

    return squint_angle


def doppler_equation(
    point, sensor_position, sensor_velocity, frequency_doppler_centroid, wavelength
):
    """Evaluate doppler equation

    Parameters
    ----------
    point : np.ndarray
        point in ECEF coordinate
    sensor_position : np.ndarray
        sensor position
    sensor_velocity : np.ndarray
        sensor velociy
    frequency_doppler_centroid : float
        doppler frequency
    wavelength : float
        sensor carrier wavelength

    Returns
    -------
    npt.ArrayLike
        residual of doppler equation
    """
    point2sensor = point - sensor_position
    distance = np.linalg.norm(point2sensor, axis=0)

    def col_wise_scalar_product(matrix_a, matrix_b):
        return np.einsum(
            "ij,ij->j", matrix_a, matrix_b
        )  # Einstein notation -- col wise dot product.

    return (
        np.divide(
            2 / wavelength * col_wise_scalar_product(point2sensor, sensor_velocity),
            distance,
        )
        - frequency_doppler_centroid
    )
