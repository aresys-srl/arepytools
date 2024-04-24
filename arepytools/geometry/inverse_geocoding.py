# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Inverse geocoding module
------------------------
"""

from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

import arepytools.geometry.inverse_geocoding_core as core
from arepytools.geometry.generalsarattitude import (
    GeneralSarAttitude,
    create_attitude_boresight_normal_curve_wrapper,
)
from arepytools.geometry.generalsarorbit import GeneralSarOrbit, GSO3DCurveWrapper
from arepytools.timing.precisedatetime import PreciseDateTime


def inverse_geocoding_attitude(
    orbit: GeneralSarOrbit,
    attitude: GeneralSarAttitude,
    ground_points: npt.ArrayLike,
    az_initial_time_guesses: Optional[Union[npt.ArrayLike, PreciseDateTime]] = None,
) -> Tuple[Union[PreciseDateTime, np.ndarray], Union[float, np.ndarray]]:
    """Perform monostatic inverse geocoding from attitude information.

    Parameters
    ----------
    orbit : GeneralSarOrbit
        general sar orbit
    attitude : GeneralSarAttitude
        general sar attitude
    ground_points : npt.ArrayLike
        ground points, in the form (3,) or (N, 3)
    az_initial_time_guesses : Union[npt.ArrayLike, PreciseDateTime], optional
        azimuth times initial guesses to limit and guide the search of solutions,
        in the form (N,) or PreciseDateTime, by default None

    Returns
    -------
    Tuple[Union[PreciseDateTime, np.ndarray], Union[float, np.ndarray]]
        azimuth times array (PreciseDateTime),
        range times array (floats)
    """

    # casting input points to numpy array
    ground_points = np.asarray(ground_points)

    # checking indexes found to select only those consistent with the input az_init_times
    if az_initial_time_guesses is not None:
        az_initial_time_guesses = np.asarray(az_initial_time_guesses)
    else:
        # computing initial guesses
        az_initial_time_guesses = inverse_geocoding_monostatic_init(
            orbit, ground_points, 0, 1
        )

    # instantiating orbit and boresight normal curves
    gso_curve = GSO3DCurveWrapper(orbit=orbit)
    gsa_boresight_normal_spline = create_attitude_boresight_normal_curve_wrapper(
        attitude=attitude
    )

    # computing actual inverse geocoding
    az_times, rng_times = core.inverse_geocoding_attitude_core(
        trajectory=gso_curve,
        boresight_normal=gsa_boresight_normal_spline,
        initial_guesses=az_initial_time_guesses,
        ground_points=ground_points,
    )

    return az_times, rng_times


def inverse_geocoding_monostatic(
    orbit: GeneralSarOrbit,
    ground_points: npt.ArrayLike,
    frequencies_doppler_centroid: Union[float, npt.ArrayLike],
    wavelength: float,
    az_initial_time_guesses: Optional[Union[PreciseDateTime, npt.ArrayLike]] = None,
) -> Tuple[Union[PreciseDateTime, np.ndarray], Union[float, np.ndarray]]:
    """Monostatic inverse geocoding computation.

    Parameters
    ----------
    orbit : GeneralSarOrbit
        general sar orbit
    ground_points : npt.ArrayLike
        earth points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    frequencies_doppler_centroid : Union[float, npt.ArrayLike]
        doppler frequencies centroid values to perform the inverse geocoding,
        in the form float or (N,).
        the number of frequency must be 1 or equal to the number of points
        provided (if more than 1). If just 1 ground point is provided, several
        frequencies can be given to compute inverse geocoding at each different frequency
        doppler centroid
    wavelength : float
        carrier signal wavelength
    az_initial_time_guesses : Union[PreciseDateTime, npt.ArrayLike], optional
        azimuth times initial guesses to limit and guide the search of solutions,
        in the form (N,) or PreciseDateTime. If None, it is automatically computed, by default None

    Returns
    -------
    Tuple[Union[PreciseDateTime, np.ndarray], Union[float, np.ndarray]]
        azimuth times array (PreciseDateTime),
        range times array (float)
    """

    ground_points = np.asarray(ground_points)

    if az_initial_time_guesses is not None:
        az_initial_time_guesses = np.asarray(az_initial_time_guesses)
    else:
        # computing an initial guess for the Newton method
        az_initial_time_guesses = inverse_geocoding_monostatic_init(
            orbit,
            ground_points,
            frequencies_doppler_centroid,
            wavelength,
        )

    # instantiating orbit 3D curve wrapper
    gso_curve = GSO3DCurveWrapper(orbit=orbit)

    # performing actual invrse geocoding monostatic
    azimuth_times, range_times = core.inverse_geocoding_monostatic_core(
        trajectory=gso_curve,
        ground_points=ground_points,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        wavelength=wavelength,
        initial_guesses=az_initial_time_guesses,
    )

    return azimuth_times, range_times


def inverse_geocoding_bistatic(
    orbit_rx: GeneralSarOrbit,
    orbit_tx: GeneralSarOrbit,
    ground_points: npt.ArrayLike,
    frequencies_doppler_centroid: Union[float, npt.ArrayLike],
    wavelength: float,
    az_initial_time_guesses: Optional[Union[PreciseDateTime, npt.ArrayLike]] = None,
) -> Tuple[Union[PreciseDateTime, np.ndarray], Union[float, np.ndarray]]:
    """Bistatic inverse geocoding computation.

    Parameters
    ----------
    orbit_rx : GeneralSarOrbit
        general sar orbit for rx sensor
    orbit_tx : GeneralSarOrbit
        general sar orbit for tx sensor
    ground_points : npt.ArrayLike
        earth points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    frequencies_doppler_centroid : Union[float, npt.ArrayLike]
        doppler frequencies centroid values to perform the inverse geocoding,
        in the form float or (N,).
        the number of frequency must be 1 or equal to the number of points
        provided (if more than 1). If just 1 ground point is provided, several
        frequencies can be given to compute inverse geocoding at each different frequency
        doppler centroid
    wavelength : float
        carrier signal wavelength
    az_initial_time_guesses : Union[PreciseDateTime, npt.ArrayLike], optional
        azimuth times initial guesses to limit and guide the search of solutions,
        in the form (N,) or PreciseDateTime. If None, it is automatically computed, by default None

    Returns
    -------
    Tuple[Union[PreciseDateTime, np.ndarray], Union[float, np.ndarray]]
        azimuth times array (PreciseDateTime),
        range times array (float)
    """

    frequencies_doppler_centroid = (
        np.asarray(frequencies_doppler_centroid)
        if not np.isscalar(frequencies_doppler_centroid)
        else frequencies_doppler_centroid
    )

    if az_initial_time_guesses is not None:
        az_initial_time_guesses = np.asarray(az_initial_time_guesses)
    else:
        # computing an initial guess for the Newton method
        az_initial_time_guesses = inverse_geocoding_bistatic_init(
            orbit_rx=orbit_rx,
            orbit_tx=orbit_tx,
            ground_points=ground_points,
            frequencies_doppler_centroid=frequencies_doppler_centroid,
            wavelength=wavelength,
        )

    # checking if orbits are overlapped
    axis_start_time = np.max([orbit_tx.time_axis_array[0], orbit_rx.time_axis_array[0]])
    axis_end_time = np.min([orbit_tx.time_axis_array[-1], orbit_rx.time_axis_array[-1]])
    axis_length = axis_end_time - axis_start_time

    if axis_length < 0:
        raise core.OrbitsNotOverlappedError

    # instantiating orbit 3D curve wrapper
    gso_curve_rx = GSO3DCurveWrapper(orbit=orbit_rx)
    gso_curve_tx = GSO3DCurveWrapper(orbit=orbit_tx)

    # performing actual invrse geocoding bistatic
    azimuth_times, range_times = core.inverse_geocoding_bistatic_core(
        trajectory_rx=gso_curve_rx,
        trajectory_tx=gso_curve_tx,
        ground_points=ground_points,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        initial_guesses=az_initial_time_guesses,
        wavelength=wavelength,
    )

    return azimuth_times, range_times


def inverse_geocoding_monostatic_init(
    orbit: GeneralSarOrbit,
    ground_points: np.ndarray,
    frequencies_doppler_centroid: Union[float, npt.ArrayLike],
    wavelength: float,
) -> Union[PreciseDateTime, np.ndarray]:
    """Function to compute azimuth initial guess for Newton method for monostatic inverse geocoding.

    In principle each input ground point could be seen several times by the sensor orbit if it contains
    multiple periods. In this case, only the first occurrence is taken, i.e. the solution corresponding
    to the first period (the smallest in terms of time).

    Parameters
    ----------
    orbit : GeneralSarOrbit
        general sar orbit
    ground_points : np.ndarray
        ground points to be inverse geocoded, in the form (3,) or (N, 3)
    frequencies_doppler_centroid : Union[float, npt.ArrayLike]
        doppler frequencies centroid values to perform the inverse geocoding,
        in the form float or (N,).
        the number of frequency must be 1 or equal to the number of points
        provided (if more than 1). If just 1 ground point is provided, several
        frequencies can be given to compute inverse geocoding at each different frequency
        doppler centroid
    wavelength : float
        carrier signal wavelength

    Returns
    -------
    Union[PreciseDateTime, np.ndarray]
        azimuth times initial guesses, one for each input point
    """
    # detecting multiple azimuth solutions
    trajectory = GSO3DCurveWrapper(orbit=orbit)
    az_initial_time_guesses = core.inverse_geocoding_monostatic_init_core(
        trajectory=trajectory,
        time_axis=orbit.time_axis_array,
        ground_points=ground_points,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        wavelength=wavelength,
    )

    # keeping only first solution for each point
    az_initial_time_guesses = np.array([g[0] for g in az_initial_time_guesses])

    if az_initial_time_guesses.size == 1 and ground_points.ndim == 1:
        az_initial_time_guesses = az_initial_time_guesses[0]

    return az_initial_time_guesses


def inverse_geocoding_bistatic_init(
    orbit_rx: GeneralSarOrbit,
    orbit_tx: GeneralSarOrbit,
    ground_points: np.ndarray,
    frequencies_doppler_centroid: Union[float, npt.ArrayLike],
    wavelength: float,
) -> Union[PreciseDateTime, np.ndarray]:
    """Function to compute azimuth initial guess for Newton method for bistatic inverse geocoding.

    Parameters
    ----------
    orbit_rx : GeneralSarOrbit
        general sar orbit for sensor rx
    orbit_tx : GeneralSarOrbit
        general sar orbit for sensor tx
    ground_points : np.ndarray
        ground points to be inverse geocoded, in the form (3,) or (N, 3)
    frequencies_doppler_centroid : Union[float, npt.ArrayLike]
        doppler frequencies centroid values to perform the inverse geocoding,
        in the form float or (N,).
        the number of frequency must be 1 or equal to the number of points
        provided (if more than 1). If just 1 ground point is provided, several
        frequencies can be given to compute inverse geocoding at each different frequency
        doppler centroid
    wavelength : float
        carrier signal wavelength

    Returns
    -------
    Union[PreciseDateTime, np.ndarray]
        azimuth times initial guesses
    """

    trajectory_rx = GSO3DCurveWrapper(orbit_rx)
    trajectory_tx = GSO3DCurveWrapper(orbit_tx)

    return core.inverse_geocoding_bistatic_init_core(
        trajectory_rx=trajectory_rx,
        trajectory_tx=trajectory_tx,
        time_axis_rx=orbit_rx.time_axis_array,
        time_axis_tx=orbit_tx.time_axis_array,
        ground_points=ground_points,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        wavelength=wavelength,
    )
