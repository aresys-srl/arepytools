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

from arepytools.geometry import conversions
from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve
from arepytools.geometry.direct_geocoding import (
    GeocodingSide,
    direct_geocoding_monostatic,
    direct_geocoding_with_look_angles,
)
from arepytools.geometry.reference_frames import ReferenceFrame, ReferenceFrameLike
from arepytools.timing.precisedatetime import PreciseDateTime


def compute_incidence_angles_from_trajectory(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    range_times: npt.ArrayLike,
    look_direction: str | GeocodingSide,
    geodetic_altitude: float | None = None,
    frequencies_doppler_centroid: npt.ArrayLike | None = None,
    carrier_wavelength: float | None = None,
) -> npt.ArrayLike:
    """Compute incidence angles from sensor trajectory (TwiceDifferentiable3DCurve compliant object).

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor trajectory compliant to the TwiceDifferentiable3DCurve protocol
    azimuth_time : PreciseDateTime
        azimuth time at which compute the incidence a angles corresponding to the input range times
    range_times : npt.ArrayLike
        range times where to compute the incidence angles, a float or a (N,) array
    look_direction : str | GeocodingSide
        side where to perform geocoding
    geodetic_altitude : float | None, optional
        the altitude over wgs84, if None is set to 0, by default None,
    frequencies_doppler_centroid : npt.ArrayLike | None, optional
        frequency_doppler_centroid value, if None is set to 0, by default None
    carrier_wavelength : float | None, optional
        carrier signal wavelength, if None is set to 1, by default None

    Returns
    -------
    npt.ArrayLike
        incidence angles in radians corresponding to the input range times at the given azimuth time
    """
    sensor_position = trajectory.evaluate(azimuth_time)
    sensor_velocity = trajectory.evaluate_first_derivatives(azimuth_time)
    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_position,
        sensor_velocities=sensor_velocity,
        range_times=range_times,
        frequencies_doppler_centroid=(
            frequencies_doppler_centroid
            if frequencies_doppler_centroid is not None
            else 0
        ),
        wavelength=carrier_wavelength if carrier_wavelength is not None else 1,
        geocoding_side=GeocodingSide(look_direction),
        geodetic_altitude=geodetic_altitude if geodetic_altitude is not None else 0,
    )
    return compute_incidence_angles(
        sensor_positions=sensor_position, points=ground_points
    )


def compute_look_angles_from_trajectory(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    range_times: npt.ArrayLike,
    look_direction: str | GeocodingSide,
    geodetic_altitude: float | None = None,
    frequencies_doppler_centroid: npt.ArrayLike | None = None,
    carrier_wavelength: float | None = None,
) -> npt.ArrayLike:
    """Compute look angles from sensor trajectory (TwiceDifferentiable3DCurve compliant object).

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor trajectory compliant to the TwiceDifferentiable3DCurve protocol
    azimuth_time : PreciseDateTime
        azimuth time at which compute the look a angles corresponding to the input range times
    range_times : npt.ArrayLike
        range times where to compute the look angles, a float or a (N,) array
    look_direction : str | GeocodingSide
        side where to perform geocoding
    geodetic_altitude : float | None, optional
        the altitude over wgs84, if None is set to 0, by default None,
    frequencies_doppler_centroid : npt.ArrayLike | None, optional
        frequency_doppler_centroid value, if None is set to 0, by default None
    carrier_wavelength : float | None, optional
        carrier signal wavelength, if None is set to 1, by default None

    Returns
    -------
    npt.ArrayLike
        look angles in radians corresponding to the input range times at the given azimuth time
    """
    sensor_position = trajectory.evaluate(azimuth_time)
    sensor_velocity = trajectory.evaluate_first_derivatives(azimuth_time)
    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_position,
        sensor_velocities=sensor_velocity,
        range_times=range_times,
        frequencies_doppler_centroid=(
            frequencies_doppler_centroid
            if frequencies_doppler_centroid is not None
            else 0
        ),
        wavelength=carrier_wavelength if carrier_wavelength is not None else 1,
        geocoding_side=GeocodingSide(look_direction),
        geodetic_altitude=geodetic_altitude if geodetic_altitude is not None else 0,
    )
    # TODO move nadir computation directly inside compute_look_angles by default (it depends only on sensor position)
    nadir = compute_nadir_from_sensor_positions(sensor_positions=sensor_position)
    return compute_look_angles(
        sensor_positions=sensor_position, nadir_directions=nadir, points=ground_points
    )


def compute_ground_velocity_from_trajectory(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    look_angles_rad: npt.ArrayLike,
    reference_frame: ReferenceFrameLike = ReferenceFrame.zero_doppler,
    geodetic_altitude: float = 0,
    averaging_interval_relative_origin: float = 0,
    averaging_interval_duration: float = 1,
    averaging_interval_num_points: int = 11,
) -> npt.ArrayLike:
    """Numerically compute the ground velocity at given look angles.

    The algorithm is based on the direct geocoding, via look angles, of points at different
    azimuth times in a averaging interval.

    Parameters
    ----------
    orbit : GeneralSarOrbit
        orbit
    time_point : PreciseDateTime
        a time point
    look_angles : npt.ArrayLike
        scalar or (N,) array like of look angles in radians
    reference_frame : ReferenceFrameLike, optional
        the reference frames in which the look angles are intended, by default ReferenceFrame.zero_doppler
    altitude_over_wgs84 : float, optional
        altitude of the points over wgs84, by default 0
    averaging_interval_relative_origin : float, optional
        averaging interval starts at
        time_point plus averaging_interval_relative_origin, by default 0
    averaging_interval_duration : float, optional
        total duration of the averaging interval, by default 1.0
    averaging_interval_num_points : int, optional
        number of time points in the averaging interval, by default 11

    Returns
    -------
    Union[float, np.ndarray]
        scalar or (N,) np array with the ground velocity
    """

    # generating the averaging time interval axis
    averaging_time_axis = (
        np.linspace(
            averaging_interval_relative_origin,
            averaging_interval_duration,
            averaging_interval_num_points,
        )
        + azimuth_time
    )
    sensor_positions = trajectory.evaluate(averaging_time_axis)
    sensor_velocities = trajectory.evaluate_first_derivatives(averaging_time_axis)

    # computing ground points at each sensor position/velocity in the selected averaging time interval, for each
    # input look angle
    look_angles = np.atleast_1d(look_angles_rad)
    ground_points = []
    for angle in look_angles:
        ground_points.append(
            direct_geocoding_with_look_angles(
                sensor_positions=sensor_positions,
                sensor_velocities=sensor_velocities,
                reference_frame=reference_frame,
                look_angles=angle,
                altitude_over_wgs84=geodetic_altitude,
            )
        )
    # computing ground velocity components (as ground points coordinates diff) for each time interval, for each
    # input look angle, and then computing their norm
    ground_velocities_norm = [
        np.linalg.norm(np.diff(g, axis=0), axis=-1) for g in ground_points
    ]
    ground_velocities = np.array(
        [
            np.sum(v, axis=-1) / averaging_interval_duration
            for v in ground_velocities_norm
        ]
    )

    return (
        ground_velocities
        if not isinstance(look_angles_rad, float)
        else ground_velocities[0]
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


def compute_nadir_from_sensor_positions(sensor_positions: np.ndarray) -> np.ndarray:
    """Compute nadir positions from sensor positions.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, with shape (3,), (N, 3)

    Returns
    -------
    np.ndarray
        nadir position, with shape (3,), (N, 3)
    """
    sensor_position_ground = conversions.xyz2llh(sensor_positions.T)
    sensor_position_ground[2] = 0.0
    sensor_position_ground = conversions.llh2xyz(sensor_position_ground).squeeze().T

    return sensor_position_ground - sensor_positions
