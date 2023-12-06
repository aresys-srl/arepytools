# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Direct geocoding module
-----------------------
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

import arepytools.constants as cst
from arepytools.geometry import conversions as conv
from arepytools.geometry.ellipsoid import WGS84, compute_line_ellipsoid_intersections
from arepytools.geometry.reference_frames import (
    ReferenceFrameLike,
    compute_rotation,
    compute_sensor_local_axis,
)


class GeocodingSide(Enum):
    """
    Geocoding side with respect to sensor velocity
    """

    RIGHT_LOOKING = "RIGHT"
    LEFT_LOOKING = "LEFT"


class NewtonMethodConvergenceError(RuntimeError):
    """Newton method could not converge to a root solution."""


class EmptyEllipsoidIntersection(RuntimeError):
    """Ellipsoid intersection cannot be found"""


class AmbiguousInputCorrelation(RuntimeError):
    """Ambiguous correlation between input in geocoding function. Operation not supported."""


def direct_geocoding_with_looking_direction(
    sensor_positions: npt.ArrayLike,
    looking_directions: npt.ArrayLike,
    *,
    altitude_over_wgs84: float = 0.0,
) -> np.ndarray:
    """Computes the ground points seen with the given direction

    The looking direction defines a line: its norm and sign do not matter.

    Based on :meth:`arepytools.geometry.ellipsoid.compute_line_ellipsoid_intersections`

    Parameters
    ----------
    sensor_positions : npt.ArrayLike
        sensor positions, in the form (3,) or (N, 3)
    looking_directions : npt.ArrayLike
        one or more vectors aligned with a looking direction, in the form (3,) or (N, 3)
    altitude_over_wgs84 : float, optional
        altitude over the WGS84 elliposid, by default 0.0

    Returns
    -------
    np.ndarray
        (3,) or (N, 3) one or more points, np.nan is a place-holder in case of impossible geocoding
    """
    inflated_ellipsoid = WGS84.inflate(altitude_over_wgs84)

    intersections = compute_line_ellipsoid_intersections(
        looking_directions,
        sensor_positions,
        inflated_ellipsoid,
    )

    points = np.empty(
        np.broadcast_shapes(np.shape(looking_directions), np.shape(sensor_positions))
    )

    if points.ndim == 1:
        intersections = (intersections,)

    for intersections_pair, point in zip(intersections, points.reshape((-1, 3))):
        point[:] = intersections_pair[0] if len(intersections_pair) > 0 else np.nan

    return points


def direct_geocoding_with_look_angles(
    sensor_positions: npt.ArrayLike,
    sensor_velocities: npt.ArrayLike,
    reference_frame: ReferenceFrameLike,
    look_angles: npt.ArrayLike,
    *,
    altitude_over_wgs84: float = 0.0,
) -> np.ndarray:
    """Computes the points at a given altitude over WGS84 ellipsoid seen with the given direction

    The looking direction defines a line: its norm and sign do not matter.

    Parameters
    ----------
    sensor_positions : npt.ArrayLike
        sensor positions, in the form (3,) or (N, 3)
    sensor_velocities : npt.ArrayLike
        sensor velocity, in the form (3,) or (N, 3)
    reference_frame : ReferenceFrameLike
        which reference frame to assume
    look_angles : npt.ArrayLike
        scalar or (N, ) one or more look angles in radians
    altitude_over_wgs84 : float, optional
        altitude over the WGS84 elliposid, by default 0.0

    Returns
    -------
    np.ndarray
        (3,) or (N, 3) points
    """
    local_axis = compute_sensor_local_axis(
        sensor_positions, sensor_velocities, reference_frame
    )

    rotation = compute_rotation(
        "YPR",
        yaw=np.zeros_like(look_angles),
        pitch=np.zeros_like(look_angles),
        roll=-np.asarray(look_angles),
    )

    pointing = np.matmul(local_axis, rotation.as_matrix())[..., 2]

    return direct_geocoding_with_looking_direction(
        sensor_positions, pointing, altitude_over_wgs84=altitude_over_wgs84
    )


def direct_geocoding_attitude(
    sensor_positions: np.ndarray,
    sensor_velocities: np.ndarray,
    antenna_reference_frames: np.ndarray,
    range_times: Union[float, np.ndarray],
    geocoding_side: Union[str, GeocodingSide],
    altitude_over_wgs84: float = 0.0,
    initial_guesses: np.ndarray = None,
) -> np.ndarray:
    """Direct geocoding using attitude information.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, in the form (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocity, in the form (3,) or (N, 3)
    antenna_reference_frames : np.ndarray
        antenna reference frame full matrix, in the form (3, 3) or (N, 3, 3)
    range_times : Union[float, np.ndarray]
        range times at which evaluate the direct geocoding, in the form (M,) or (1, M)
    geocoding_side : Union[str, GeocodingSide]
        side where to perform geocoding
    altitude_over_wgs84 : float, optional
        altitude over the wgs84 ellipsoid model, by default 0.0


    Returns
    -------
    np.ndarray
        matrix with the computed direct geocoding with proper dimension according to inputs

    Raises
    ------
    EmptyEllipsoidIntersection
        no line of sight intersection with WGS84 ellipsoid found
    """

    # managing position, velocity and antenna frame
    sensor_positions = np.asarray(sensor_positions)
    sensor_velocities = np.asarray(sensor_velocities)
    antenna_reference_frames = np.asarray(antenna_reference_frames)
    geocoding_side = GeocodingSide(geocoding_side)

    normalized_velocity = np.linalg.norm(sensor_velocities, axis=-1, keepdims=True)

    scaled_arf_velocities = normalized_velocity * antenna_reference_frames[..., :, 0]

    if initial_guesses is None:
        initial_guesses = direct_geocoding_monostatic_attitude_init(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            geocoding_side=geocoding_side,
            antenna_reference_frames=antenna_reference_frames,
        )

    return direct_geocoding_monostatic(
        sensor_positions=sensor_positions,
        sensor_velocities=scaled_arf_velocities,
        initial_guesses=initial_guesses,
        range_times=range_times,
        geocoding_side=geocoding_side,
        geodetic_altitude=altitude_over_wgs84,
        frequencies_doppler_centroid=0,
        wavelength=1,
    )


def direct_geocoding_monostatic(
    sensor_positions: npt.ArrayLike,
    sensor_velocities: npt.ArrayLike,
    range_times: Union[float, npt.ArrayLike],
    frequencies_doppler_centroid: Union[float, npt.ArrayLike],
    wavelength: float,
    geocoding_side: Union[str, GeocodingSide],
    geodetic_altitude: float,
    initial_guesses: np.ndarray = None,
) -> np.ndarray:
    """Perform direct geocoding for monostatic sensor.

    Parameters
    ----------
    sensor_positions : npt.ArrayLike
        position of the sensor, shape (3,) or (N, 3)
    sensor_velocities : npt.ArrayLike
        velocity of the sensor, shape (3,) or (N, 3)
    range_times : Union[float, npt.ArrayLike]
        range times, float or (M,)
    frequencies_doppler_centroid : Union[float, npt.ArrayLike]
        frequency_doppler_centroid value, single value or array (same length of range times),
        if a single value is passed and there is more than 1 range times, it is broadcasted to all of them
    wavelength : float
        carrier signal wavelength
    geocoding_side : Union[str, GeocodingSide]
        side where to perform geocoding
    geodetic_altitude : float
        the altitude over wgs84
    initial_guesses : np.ndarray, optional
        initial guess for newton method. If not provided a guess will be computed, by default None

    Returns
    -------
    np.ndarray
        geocoded position for each input time and position value

    Raises
    ------
    AmbiguousInputCorrelation
        if inputs shapes are ambigous to match, this error is raised
    """

    # input vectorization
    sensor_positions = np.asarray(sensor_positions)
    sensor_velocities = np.asarray(sensor_velocities)
    initial_guesses = (
        np.asarray(initial_guesses) if initial_guesses is not None else None
    )
    range_times = (
        np.asarray(range_times) if not isinstance(range_times, float) else range_times
    )
    geocoding_side = GeocodingSide(geocoding_side)

    # computation of initial guesses, if not provided
    if initial_guesses is None:
        # computing mid range distance
        average_input_range = np.median(range_times) * cst.LIGHT_SPEED / 2
        initial_guesses = direct_geocoding_monostatic_init(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            range_distance=average_input_range,
            geocoding_side=geocoding_side,
        )

    # direct geocoding monostatic core
    ground_points = _direct_geocoding_monostatic_core(
        initial_guesses=initial_guesses,
        sensor_positions=sensor_positions,
        sensor_velocities=sensor_velocities,
        range_times=range_times,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        wavelength=wavelength,
        geodetic_altitude=geodetic_altitude,
    )

    return ground_points


def direct_geocoding_bistatic(
    sensor_positions_rx: npt.ArrayLike,
    sensor_velocities_rx: npt.ArrayLike,
    sensor_positions_tx: npt.ArrayLike,
    sensor_velocities_tx: npt.ArrayLike,
    range_times: Union[float, npt.ArrayLike],
    frequencies_doppler_centroid: Union[float, npt.ArrayLike],
    wavelength: float,
    geocoding_side: Union[str, GeocodingSide],
    geodetic_altitude: float,
    initial_guesses: npt.ArrayLike = None,
) -> np.ndarray:
    """Perform direct geocoding for bistatic sensors.

    Parameters
    ----------
    sensor_positions_rx : npt.ArrayLike
        position of the sensor rx, in the form (3,) or (N, 3)
    sensor_velocities_rx : npt.ArrayLike
        velocity of the sensor rx, in the form (3,) or (N, 3)
    sensor_positions_tx : npt.ArrayLike
        position of the sensor tx, in the form (3,) or (M, 3), where M is the number of range times
    sensor_velocities_tx : npt.ArrayLike
        velocity of the sensor tx, in the form (3,) or (M, 3), where M is the number of range times
    range_times : Union[float, npt.ArrayLike]
        range times where to evaluate the direct geocoding, in the form float or (M,)
    frequencies_doppler_centroid : Union[float, npt.ArrayLike]
        frequency doppler centroid values, in the form float or (M,). If a single value is given but multiple
        range times are provided, it is automatically broadcasted to all of them
    wavelength : float
        carrier signal wavelength
    geocoding_side : Union[str, GeocodingSide]
        side where to perform geocoding
    geodetic_altitude : float
        altitude with respect to the WGS84 ellipsoid
    initial_guesses : npt.ArrayLike, optional
        initial guess for Newton method. If not provided a guess will be computed, by default None

    Returns
    -------
    np.ndarray
        ground points for each input time and position rx value
    """

    # input vectorization
    sensor_positions_rx = np.asarray(sensor_positions_rx)
    sensor_velocities_rx = np.asarray(sensor_velocities_rx)
    sensor_positions_tx = np.asarray(sensor_positions_tx)
    sensor_velocities_tx = np.asarray(sensor_velocities_tx)
    geocoding_side = GeocodingSide(geocoding_side)

    # Optional initial guess
    if initial_guesses is None:
        # computing mid range distance
        average_input_range = np.median(range_times) * cst.LIGHT_SPEED / 2
        initial_guesses = direct_geocoding_monostatic_init(
            sensor_positions=sensor_positions_rx,
            sensor_velocities=sensor_velocities_rx,
            range_distance=average_input_range,
            geocoding_side=geocoding_side,
        )

    # direct geocoding bistatic core
    ground_points = _direct_geocoding_bistatic_core(
        initial_guesses=initial_guesses,
        sensor_positions_rx=sensor_positions_rx,
        sensor_velocities_rx=sensor_velocities_rx,
        sensor_positions_tx=sensor_positions_tx,
        sensor_velocities_tx=sensor_velocities_tx,
        range_times=range_times,
        wavelength=wavelength,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        geodetic_altitude=geodetic_altitude,
    )

    return ground_points


def _direct_geocoding_monostatic_core(
    sensor_positions: np.ndarray,
    sensor_velocities: np.ndarray,
    range_times: Union[float, npt.ArrayLike],
    frequencies_doppler_centroid: Union[float, npt.ArrayLike],
    wavelength: float,
    geodetic_altitude: float,
    initial_guesses: np.ndarray,
) -> np.ndarray:
    """Computation of direct geocoding for monostatic systems.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor position array, in the form (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocity array, in the form (3,) or (N, 3)
    range_times : Union[float, npt.ArrayLike]
        range times at which evaluate the geocoding equation, in the form float or (M,)
    frequencies_doppler_centroid : Union[float, npt.ArrayLike]
        frequency_doppler_centroid value, single value or array (same length of range times),
        if a single value is passed and there is more than 1 range times, it is broadcasted to all of them
    wavelength : float
        carrier signal wavelength
    geodetic_altitude : float
        geodetic altitude with respect to WGS84 ellipsoid
    initial_guesses : np.ndarray
        initial guess for the newton method, in the form (3,) or (N, 3) or (M, 3) if 1 position and
        M range times

    Returns
    -------
    np.ndarray
        ground points, solution to the Newton method for direct geocoding

    Raises
    ------
    AmbiguousInputCorrelation
        if inputs shapes are ambigous to match, this error is raised
    """

    range_times = (
        np.asarray(range_times)
        if not isinstance(range_times, float)
        else np.asarray([range_times])
    )

    try:
        frequencies_doppler_centroid = np.broadcast_to(
            frequencies_doppler_centroid, range_times.shape
        )
    except ValueError as exc:
        raise AmbiguousInputCorrelation(
            f"frequencies {frequencies_doppler_centroid.shape} != range times {range_times.shape}"
        ) from exc

    try:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)
    except ValueError as exc:
        raise AmbiguousInputCorrelation(
            f"sensor position {sensor_positions.shape} != sensor velocities {sensor_velocities.shape}"
        ) from exc

    try:
        initial_guesses = np.broadcast_to(initial_guesses, sensor_positions.shape)
    except ValueError as exc:
        raise AmbiguousInputCorrelation(
            f"sensor position {sensor_positions.shape} != initial guesses {initial_guesses.shape}"
        ) from exc

    if not sensor_positions.shape == sensor_velocities.shape == initial_guesses.shape:
        # case: different shapes between inputs
        raise AmbiguousInputCorrelation(
            f"Mismatch between input shapes: pos {sensor_positions.shape},"
            + f"vel {sensor_velocities.shape}, guess {initial_guesses.shape}"
        )

    one_size_array_flag = 0
    if sensor_positions.ndim == 2 and sensor_positions.size / 3 == 1:
        one_size_array_flag = 1

    ground_points = np.zeros((sensor_positions.size // 3, range_times.size, 3))
    for id_rng, rng_freq in enumerate(zip(range_times, frequencies_doppler_centroid)):
        ground_points[..., id_rng, :] = _newton_for_direct_geocoding_monostatic(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            initial_guesses=initial_guesses,
            range_time=rng_freq[0],
            frequency_doppler_centroid=rng_freq[1],
            geodetic_altitude=geodetic_altitude,
            wavelength=wavelength,
        )

    return (
        ground_points.squeeze()
        if not one_size_array_flag
        else ground_points.squeeze(axis=0)
    )


def _direct_geocoding_bistatic_core(
    sensor_positions_rx: np.ndarray,
    sensor_velocities_rx: np.ndarray,
    sensor_positions_tx: np.ndarray,
    sensor_velocities_tx: np.ndarray,
    range_times: Union[float, np.ndarray],
    frequencies_doppler_centroid: Union[float, np.ndarray],
    wavelength: float,
    geodetic_altitude: float,
    initial_guesses: np.ndarray,
) -> np.ndarray:
    """Computation of direct geocoding for bistatic systems.

    Parameters
    ----------
    sensor_positions_rx : npt.ArrayLike
        position of the sensor rx, in the form (3,) or (N, 3)
    sensor_velocities_rx : npt.ArrayLike
        velocity of the sensor rx, in the form (3,) or (N, 3)
    sensor_positions_tx : npt.ArrayLike
        position of the sensor tx, in the form (3,) or (M, 3), where M is the number of range times
    sensor_velocities_tx : npt.ArrayLike
        velocity of the sensor tx, in the form (3,) or (M, 3), where M is the number of range times
    range_times : Union[float, np.ndarray]
        range times where to evaluate the direct geocoding, in the form float or (M,)
    frequencies_doppler_centroid : Union[float, np.ndarray]
        frequency doppler centroid values, in the form float or (M,). If a single value is given but multiple
        range times are provided, it is automatically broadcasted to all of them
    wavelength : float
        carrier signal wavelength
    geodetic_altitude : float
        altitude with respect to the WGS84 ellipsoid
    initial_guesses : np.ndarray
        initial guess for Newton method, in the form (3,), (N, 3) or (M, 3) if just 1 poisition and M range times

    Returns
    -------
    np.ndarray
        ground points for each input time and position rx value

    Raises
    ------
    AmbiguousInputCorrelation
        if inputs shapes are ambigous to match, this error is raised
    """

    range_times = (
        np.asarray(range_times)
        if not isinstance(range_times, float)
        else np.asarray([range_times])
    )
    try:
        frequencies_doppler_centroid = np.broadcast_to(
            frequencies_doppler_centroid, range_times.shape
        )
    except ValueError as excp:
        raise AmbiguousInputCorrelation(
            f"frequencies {frequencies_doppler_centroid.shape} != range times {range_times.shape}"
        ) from excp

    one_size_array_flag = 0
    if (sensor_positions_rx.ndim == 2 and sensor_positions_rx.size / 3 == 1) or (
        sensor_positions_tx.ndim == 2
        and sensor_positions_tx.size / 3 == 1
        and sensor_positions_rx.size / 3 == 1
    ):
        one_size_array_flag = 1

    if (
        sensor_positions_tx.ndim == sensor_velocities_tx.ndim == 1
        and sensor_positions_tx.size // 3 == 1
    ):
        sensor_positions_tx = sensor_positions_tx.reshape(1, sensor_positions_tx.size)
        sensor_velocities_tx = sensor_velocities_tx.reshape(
            1, sensor_velocities_tx.size
        )

    ground_points = np.zeros((sensor_positions_rx.size // 3, range_times.size, 3))
    looping_items = zip(
        range_times,
        frequencies_doppler_centroid,
        sensor_positions_tx,
        sensor_velocities_tx,
    )
    for id_rng, items in enumerate(looping_items):
        ground_points[..., id_rng, :] = _newton_for_direct_geocoding_bistatic(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            initial_guesses=initial_guesses,
            sensor_position_tx=items[2],
            sensor_velocity_tx=items[3],
            range_time=items[0],
            frequency_doppler_centroid=items[1],
            geodetic_altitude=geodetic_altitude,
            wavelength=wavelength,
        )

    return (
        ground_points.squeeze()
        if not one_size_array_flag
        else ground_points.squeeze(axis=0)
    )


def direct_geocoding_monostatic_init(
    sensor_positions: np.ndarray,
    sensor_velocities: np.ndarray,
    range_distance: float,
    geocoding_side: Union[str, GeocodingSide],
) -> np.ndarray:
    """Computation of initial guesses for direct geocoding monostatic.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, in the form (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocity, in the form (3,) or (N, 3)
    range_distance : float
        range distance
    geocoding_side : Union[str, GeocodingSide]
        side where to perform geocoding

    Returns
    -------
    np.ndarray
        initial guess ground points

    Raises
    ------
    RuntimeError
        if range distance not compatibile with sensor position and earth radius
    """

    one_size_array_flag = 0
    if sensor_velocities.ndim == sensor_positions.ndim == 1:
        one_size_array_flag = 1

    if sensor_positions.ndim < sensor_velocities.ndim:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)

    geocoding_side = GeocodingSide(geocoding_side)
    geocoding_side_factor = 1 if geocoding_side == GeocodingSide.RIGHT_LOOKING else -1

    sensor_position_norm = np.linalg.norm(sensor_positions, axis=-1, keepdims=True)
    llh_sat = conv.xyz2llh(sensor_positions.T)
    xyz_sat = conv.llh2xyz(
        np.array([llh_sat[0], llh_sat[1], np.zeros(llh_sat.shape[1])])
    )
    earth_radius = np.linalg.norm(xyz_sat.T, axis=-1, keepdims=True)

    # check earth radius vs range compatibility
    if any(range_distance < sensor_position_norm - earth_radius):
        raise RuntimeError("Cannot find initial guess for direct geocoding")

    u_x = sensor_positions / sensor_position_norm
    u_y = np.cross(sensor_positions, sensor_velocities)
    u_y = u_y / np.linalg.norm(u_y, axis=-1, keepdims=True)
    u_z = np.cross(u_x, u_y)

    # x-coordinate
    coords = (sensor_position_norm**2 + earth_radius**2 - range_distance**2) / (
        2 * sensor_position_norm
    )

    # circle radius
    circle_radius = np.sqrt(earth_radius**2 - coords**2)

    # Project velocity on ref frame
    v_x = np.sum(sensor_velocities * u_x, axis=-1)
    v_z = np.sum(sensor_velocities * u_z, axis=-1)

    # Find first solution
    z_solution = (sensor_position_norm - coords).T * v_x / v_z
    y_solution = np.sqrt(circle_radius.T**2 - z_solution**2)

    # inverting y solution by look sign
    y_solution[y_solution * geocoding_side_factor > 0] *= -1

    init_guess = coords * u_x + y_solution.T * u_y + z_solution.T * u_z

    return init_guess if not one_size_array_flag else init_guess.squeeze()


def direct_geocoding_monostatic_attitude_init(
    sensor_positions: np.ndarray,
    sensor_velocities: np.ndarray,
    antenna_reference_frames: np.ndarray,
    geocoding_side: Union[str, GeocodingSide],
    perturbation_for_nadir_geom: bool = False,
    perturbation_length: float = 100.0,
) -> np.ndarray:
    """Computation of initial guesses for direct geocoding monostatic with attitude.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, in the form (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocity, in the form (3,) or (N, 3)
    antenna_reference_frames : np.ndarray
        antenna reference frame full matrix, in the form (3, 3) or (N, 3, 3)
    geocoding_side : Union[str, GeocodingSide]
        side where to perform geocoding
    perturbation_for_nadir_geom : bool, optional
        boolean flag to enable the perturbation of the initial guess by adding perturbation length, by default False
    perturbation_length : float, optional
        perturbation length along nadir direction in meters, by default 100.0

    Returns
    -------
    np.ndarray
        initial guess ground points

    Raises
    ------
    RuntimeError
        if no solutions have been found after intersection with WGS84 ellipsoid
    """

    geocoding_side = GeocodingSide(geocoding_side)
    boresight_directions = -antenna_reference_frames[..., :, 2].copy()

    if sensor_positions.ndim < sensor_velocities.ndim:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)

    # computing the intersection with the ellipsoid (WGS84 model)
    solutions = compute_line_ellipsoid_intersections(
        ellipsoid=WGS84,
        line_origins=sensor_positions.copy(),
        line_directions=boresight_directions.copy(),
    )
    if not solutions:
        raise RuntimeError(
            "Cannot find initial guess: "
            + "cannot find intersection between antenna bore sight and WGS84 ellipsoid"
        )
    ground_point_guesses = (
        solutions[0]
        if isinstance(solutions[0], np.ndarray)
        else np.array([s[0] for s in solutions])
    )

    if perturbation_for_nadir_geom:
        # moving the initial guess in the direction orthogonal to Nadir and sensor velocity
        perturbation_dir = np.cross(sensor_velocities, sensor_positions)
        norm_perturbation_dir = np.linalg.norm(perturbation_dir, axis=-1, keepdims=True)
        perturbation_dir_norm = perturbation_dir / norm_perturbation_dir

        if geocoding_side == GeocodingSide("LEFT"):
            # invert direction
            perturbation_dir_norm *= -1

        ground_point_guesses = (
            ground_point_guesses + perturbation_dir_norm * perturbation_length
        )

    return ground_point_guesses


def _newton_for_direct_geocoding_bistatic(
    sensor_positions_rx: np.ndarray,
    sensor_velocities_rx: np.ndarray,
    initial_guesses: np.ndarray,
    sensor_position_tx: np.ndarray,
    sensor_velocity_tx: np.ndarray,
    range_time: float,
    frequency_doppler_centroid: float,
    wavelength: float,
    geodetic_altitude: float,
    max_iter: int = 8,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Newton solving method for direct geocoding bistatic.

    Parameters
    ----------
    sensor_positions_rx : np.ndarray
        sensor rx poisition array, in the form (3,) or (N,3)
    sensor_velocities_rx : np.ndarray
        sensor rx velocities array, in the form (3,) or (N,3)
    initial_guesses : np.ndarray
        initial guesses array, in the form (3,) or (N,3)
    sensor_position_tx : np.ndarray
        sensor tx poisition, in the form (3,)
    sensor_velocity_tx : np.ndarray
        sensor tx velocity, in the form (3,)
    range_time : float
        range time at which compute the geocoding equation
    frequency_doppler_centroid : float
        frequency doppler centroid at which compute the geocoding equation
    wavelength : float
        carrier signal wavelength
    geodetic_altitude : float
        geodetic altitude with respect to WGS84 ellipse
    max_iter : int, optional
        maximum iterations for Newton method, by default 8
    tolerance : float, optional
        tolerance below which assert Newton convergence in meters, by default 1E-5

    Returns
    -------
    np.ndarray
        earth points at a given range value

    Raises
    ------
    NewtonMethodConvergenceError
        raised if Newton method did not converge after max_iterations
    """

    tolerance_squared = tolerance * tolerance

    # variables and constants computation
    range_distance_square = (cst.LIGHT_SPEED * range_time) ** 2  # two-way distance
    geoid_r_min = WGS84.semi_minor_axis + geodetic_altitude
    geoid_r_max = WGS84.semi_major_axis + geodetic_altitude
    r_ep2 = geoid_r_min**2
    r_ee2 = geoid_r_max**2

    # input arguments array conversion
    sensor_positions_rx = np.atleast_2d(sensor_positions_rx)
    sensor_velocities_rx = np.atleast_2d(sensor_velocities_rx)
    sensor_position_tx = np.atleast_2d(sensor_position_tx)
    sensor_velocity_tx = np.atleast_2d(sensor_velocity_tx)
    ground_points_guess = np.asarray(initial_guesses).copy()

    # newton method for direct geocoding
    for _ in range(max_iter):
        # first sensor data
        line_of_sight_rx = sensor_positions_rx - ground_points_guess
        distance_square_rx = np.sum(line_of_sight_rx * line_of_sight_rx, axis=-1)
        distance_rx = np.sqrt(distance_square_rx)
        los_vel_product_rx = np.sum(sensor_velocities_rx * line_of_sight_rx, axis=-1)

        # second sensor data
        line_of_sight_tx = sensor_position_tx - ground_points_guess
        distance_square_tx = np.sum(line_of_sight_tx * line_of_sight_tx, axis=-1)
        distance_tx = np.sqrt(distance_square_tx)
        los_vel_product_tx = np.sum(sensor_velocity_tx * line_of_sight_tx, axis=-1)

        # range equation
        distance = distance_rx + distance_tx
        range_equation = distance**2 - range_distance_square
        grad_range_equation = (
            -2
            * distance[:, np.newaxis]
            * (
                line_of_sight_rx / distance_rx[:, np.newaxis]
                + line_of_sight_tx / distance_tx[:, np.newaxis]
            )
        )

        # doppler equations
        doppler_equation_rx, grad_doppler_equation_rx = _doppler_equation(
            wavelength=wavelength,
            pv_scalar=los_vel_product_rx,
            distance=distance_rx,
            frequency_doppler_centroid=frequency_doppler_centroid,
            sat_velocity=sensor_velocities_rx,
            sat2point=line_of_sight_rx,
        )
        doppler_equation_tx, grad_doppler_equation_tx = _doppler_equation(
            wavelength=wavelength,
            pv_scalar=los_vel_product_tx,
            distance=distance_tx,
            frequency_doppler_centroid=frequency_doppler_centroid,
            sat_velocity=sensor_velocity_tx,
            sat2point=line_of_sight_tx,
        )

        # assembling doppler equations and their gradients
        doppler_equation = (doppler_equation_rx + doppler_equation_tx) / 2
        grad_doppler_equation = (
            grad_doppler_equation_rx + grad_doppler_equation_tx
        ) / 2

        # assembling system of equations to be solved using Newton method
        functions_to_be_solved = [
            range_equation,
            _ellipse_equation(ground_points_guess, r_ee2, r_ep2),
            doppler_equation,
        ]
        functions_jacobians = [
            [
                grad_range_equation[..., k],
                _der_ellipse_equation_xi(ground_points_guess, k, r_ee2, r_ep2),
                grad_doppler_equation[..., k],
            ]
            for k in range(3)
        ]

        delta_err = -_inv_3x3_transpose(
            functions_jacobians, functions_to_be_solved
        ).squeeze()
        ground_points_guess = ground_points_guess + delta_err.T

        err_for_convergence = np.dot(delta_err, delta_err.T)
        if np.max(np.abs(err_for_convergence)) <= tolerance_squared:
            break
    else:
        raise NewtonMethodConvergenceError(
            f"Newton did not converge: maximum number of iterations {max_iter} reached. Residual error {delta_err}"
        )

    return ground_points_guess


def _newton_for_direct_geocoding_monostatic(
    sensor_positions: npt.ArrayLike,
    sensor_velocities: npt.ArrayLike,
    initial_guesses: npt.ArrayLike,
    range_time: float,
    frequency_doppler_centroid: float,
    wavelength: float,
    geodetic_altitude: float,
    max_iter: int = 8,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Newton solving method for direct geocoding monostatic.

    Parameters
    ----------
    sensor_positions : npt.ArrayLike
        sensor poisition array, in the form (3,) or (N,3)
    sensor_velocities : npt.ArrayLike
        sensor velocities array, in the form (3,) or (N,3)
    initial_guesses : npt.ArrayLike
        initial guesses array, in the form (3,) or (N,3)
    range_time : float
        range time at which compute the geocoding equation
    frequency_doppler_centroid : float
        frequency doppler centroid at which compute the geocoding equation
    wavelength : float
        carrier signal wavelength
    geodetic_altitude : float
        geodetic altitude with respect to WGS84 ellipse
    max_iter : int, optional
        maximum iterations for Newton method, by default 8
    tolerance : float, optional
        tolerance below which assert Newton convergence in meters, by default 1E-5

    Returns
    -------
    np.ndarray
        earth points at a given range value

    Raises
    ------
    NewtonMethodConvergenceError
        raised if Newton method did not converge after max_iterations
    """

    tolerance_squared = tolerance * tolerance

    # variables and constants computation
    range_distance_square = (cst.LIGHT_SPEED * range_time / 2.0) ** 2
    geoid_r_min = WGS84.semi_minor_axis + geodetic_altitude
    geoid_r_max = WGS84.semi_major_axis + geodetic_altitude
    r_ep2 = geoid_r_min**2
    r_ee2 = geoid_r_max**2

    # input arguments array conversion
    sensor_positions = np.asarray(sensor_positions)
    sensor_velocities = np.asarray(sensor_velocities)
    ground_points_guess = np.asarray(initial_guesses).copy()

    array_size_one_flag = 0
    if ground_points_guess.ndim == sensor_positions.ndim == sensor_velocities.ndim == 1:
        array_size_one_flag = 1
        ground_points_guess = ground_points_guess.reshape(1, ground_points_guess.size)
        sensor_positions = sensor_positions.reshape(1, sensor_positions.size)
        sensor_velocities = sensor_velocities.reshape(1, sensor_velocities.size)

    # newton method for direct geocoding
    for _ in range(max_iter):
        line_of_sight = sensor_positions - ground_points_guess
        distance_square = np.sum(line_of_sight * line_of_sight, axis=-1)
        distance = np.sqrt(distance_square)
        los_vel_product = np.sum(sensor_velocities * line_of_sight, axis=-1)

        range_equation = distance_square - range_distance_square
        grad_range_equation = -2 * line_of_sight

        doppler_equation, grad_doppler_equation = _doppler_equation(
            pv_scalar=los_vel_product,
            sat2point=line_of_sight,
            sat_velocity=sensor_velocities,
            distance=distance,
            wavelength=wavelength,
            frequency_doppler_centroid=frequency_doppler_centroid,
        )

        # assembling system of equations to be solved using Newton method
        functions_to_be_solved = [
            range_equation,
            _ellipse_equation(ground_points_guess, r_ee2, r_ep2),
            doppler_equation,
        ]
        functions_jacobians = [
            [
                grad_range_equation[..., k],
                _der_ellipse_equation_xi(ground_points_guess, k, r_ee2, r_ep2),
                grad_doppler_equation[..., k],
            ]
            for k in range(3)
        ]

        delta_err = (
            -_inv_3x3_transpose(functions_jacobians, functions_to_be_solved).squeeze().T
        )
        ground_points_guess = ground_points_guess + delta_err

        err_for_convergence = np.sum(delta_err * delta_err, axis=-1)
        if np.max(np.abs(err_for_convergence)) <= tolerance_squared:
            break
    else:
        raise NewtonMethodConvergenceError(
            f"Newton did not converge: maximum number of iterations {max_iter} reached. Residual error {delta_err}"
        )

    return (
        ground_points_guess
        if not array_size_one_flag
        else ground_points_guess.squeeze()
    )


def _inv_3x3_transpose(jac: np.ndarray, func: np.ndarray) -> np.ndarray:
    """Performing inverse of 3x3 matrix using explicit form.

    Parameters
    ----------
    jac : np.ndarray
        jacobians array
    func : np.ndarray
        functions array

    Returns
    -------
    np.ndarray
        inverse of input func matrix
    """
    det = (
        +jac[0][0] * (jac[2][2] * jac[1][1] - jac[2][1] * jac[1][2])
        - jac[1][0] * (jac[2][2] * jac[0][1] - jac[2][1] * jac[0][2])
        + jac[2][0] * (jac[1][2] * jac[0][1] - jac[1][1] * jac[0][2])
    )

    x_val = (
        func[0] * (jac[1][1] * jac[2][2] - jac[2][1] * jac[1][2])
        + func[1] * (jac[2][0] * jac[1][2] - jac[1][0] * jac[2][2])
        + func[2] * (jac[1][0] * jac[2][1] - jac[2][0] * jac[1][1])
    )

    y_val = (
        func[0] * (jac[2][1] * jac[0][2] - jac[0][1] * jac[2][2])
        + func[1] * (jac[0][0] * jac[2][2] - jac[2][0] * jac[0][2])
        + func[2] * (jac[2][0] * jac[0][1] - jac[0][0] * jac[2][1])
    )

    z_val = (
        func[0] * (jac[0][1] * jac[1][2] - jac[1][1] * jac[0][2])
        + func[1] * (jac[1][0] * jac[0][2] - jac[0][0] * jac[1][2])
        + func[2] * (jac[0][0] * jac[1][1] - jac[1][0] * jac[0][1])
    )

    return np.asarray([x_val, y_val, z_val]) / det


def _ellipse_equation(coords: np.ndarray, r_ee2: float, r_ep2: float) -> float:
    """3D Ellipse generic equation.

    Parameters
    ----------
    x : np.ndarray
        x, y, z coordinates array where to evaluate the ellipse
    r_ee2 : float
        radius square along x and y directions
    r_ep2 : float
        radius square along z direction

    Returns
    -------
    float
        value of the ellipse at the input coordinate
    """
    return (
        (coords[..., 0] * coords[..., 0] + coords[..., 1] * coords[..., 1]) / r_ee2
        + coords[..., 2] * coords[..., 2] / r_ep2
        - 1.0
    )


def _der_ellipse_equation_xi(
    coords: np.ndarray, i_coord: int, r_ee2: float, r_ep2: float
) -> float:
    """Derivative of ellipse equation.

    Parameters
    ----------
    x : np.ndarray
        x, y, z array coordinate where to evaluate the derivative
    i_coord : int
        direction index where to evaluate the derivative
    r_ee2 : float
        radius square along x and y directions
    r_ep2 : float
        radius square along z direction

    Returns
    -------
    float
        derivative value along the selected direction at the selected coordinate
    """

    radius_square = r_ee2 if i_coord < 2 else r_ep2

    return 2 * coords[..., i_coord] / radius_square


def _doppler_equation(
    wavelength: float,
    pv_scalar: float,
    distance: float,
    frequency_doppler_centroid: float,
    sat_velocity: np.ndarray,
    sat2point: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Doppler equation solver.

    Parameters
    ----------
    wavelength : float
        carrier signal wavelength
    pv_scalar : float
        scalar product between sensor velocity and line of sight
    distance : float
        ground point - sensor distance
    frequency_doppler_centroid : float
        frequency doppler centroid
    sat_velocity : np.ndarray
        sensor velocity
    sat2point : np.ndarray
        line of sight

    Returns
    -------
    tuple[float, np.ndarray]
        doppler equation solution,
        doppler equation gradient
    """

    c_factor = 2.0 / wavelength / distance
    doppler_equation = c_factor * pv_scalar + frequency_doppler_centroid
    norm_pv = pv_scalar / distance**2
    grad_doppler_equation = (c_factor * (-sat_velocity + (norm_pv * sat2point.T).T).T).T
    return doppler_equation, grad_doppler_equation
