# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
General SAR orbit module
------------------------
"""

from __future__ import annotations

import functools
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.optimize

import arepytools.geometry.inverse_geocoding_core as inverse_core
from arepytools.geometry import conversions
from arepytools.geometry._interpolator import GeometryInterpolator
from arepytools.geometry.direct_geocoding import (
    direct_geocoding_bistatic,
    direct_geocoding_monostatic,
    direct_geocoding_with_look_angles,
)
from arepytools.geometry.geometric_functions import (
    compute_incidence_angles,
    compute_look_angles,
    doppler_equation,
)
from arepytools.geometry.reference_frames import ReferenceFrame, ReferenceFrameLike
from arepytools.io.metadata import StateVectors
from arepytools.math import axis as are_ax
from arepytools.timing.precisedatetime import PreciseDateTime


class _MISSING_TYPE:
    pass


MISSING = _MISSING_TYPE()


class GSO3DCurveWrapper:
    """Wrapper over General Sar Orbit to manage a 3D curve using directly its methods"""

    def __init__(self, orbit: GeneralSarOrbit) -> None:
        self.orbit = orbit

    def evaluate(
        self, coordinates: Union[PreciseDateTime, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate x, y, z polynomial at given times.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, npt.ArrayLike]
            time points

        Returns
        -------
        np.ndarray
            values of the polynomials at given input times (N, 3)
        """
        if isinstance(coordinates, np.ndarray) and coordinates.ndim == 0:
            coordinates = coordinates.item()
        return self.orbit.get_position(time_points=coordinates).T.reshape(
            np.shape(coordinates) + (3,)
        )

    def evaluate_first_derivatives(
        self, coordinates: Union[PreciseDateTime, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate x, y, z polynomial first derivatives at given times.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, npt.ArrayLike]
            time points

        Returns
        -------
        np.ndarray
            values of the polynomials first derivatives at given input times (N, 3)
        """
        if isinstance(coordinates, np.ndarray) and coordinates.ndim == 0:
            coordinates = coordinates.item()
        return self.orbit.get_velocity(time_points=coordinates).T.reshape(
            np.shape(coordinates) + (3,)
        )

    def evaluate_second_derivatives(
        self, coordinates: Union[PreciseDateTime, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate x, y, z polynomial second derivatives at given times.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, npt.ArrayLike]
            time points

        Returns
        -------
        np.ndarray
            values of the polynomials second derivatives at given input times (N, 3)
        """
        if isinstance(coordinates, np.ndarray) and coordinates.ndim == 0:
            coordinates = coordinates.item()
        return self.orbit.get_acceleration(time_points=coordinates).T.reshape(
            np.shape(coordinates) + (3,)
        )


class GeneralSarOrbit:
    """General SAR orbit"""

    _MINIMUM_NUMBER_OF_DATA_POINTS = GeometryInterpolator.get_min_num_of_data_points()

    @classmethod
    def get_minimum_number_of_data_points(cls) -> int:
        """Return the required minimum number of orbit data points

        Returns
        -------
        int
            minimum number of required data points
        """
        return cls._MINIMUM_NUMBER_OF_DATA_POINTS

    @property
    def t0(self) -> PreciseDateTime:
        """Origin of the time axis"""
        assert isinstance(self._time_axis.start, PreciseDateTime)
        return self._time_axis.start

    @property
    def dt(self) -> float:
        """Time axis step (if applicable)"""
        if isinstance(self._time_axis, are_ax.RegularAxis):
            return self._time_axis.step
        raise RuntimeError(
            "Time step is not available for orbits constructed with non-regular time axis"
        )

    @property
    def n(self) -> int:
        """Number of orbit data points"""
        return self._time_axis.size

    @property
    def position_sv(self) -> npt.NDArray[np.floating]:
        """State vectors as 1D array of size 3N"""
        return self._state_vectors.reshape((self.n * 3,), order="F")

    @property
    def time_axis_array(self) -> np.ndarray:
        """Time axis as array of time points"""
        return self._time_axis.get_array()

    @property
    def interpolator(self):
        """Geometry interpolator object"""
        return self._interpolator

    @property
    def anx_times(self) -> Optional[np.ndarray]:
        """ANX times"""
        return self._anx_times

    @property
    def anx_positions(self) -> Optional[npt.NDArray[np.floating]]:
        """ANX positions"""
        return self._anx_positions

    def __init__(
        self,
        time_axis: Union[np.ndarray, are_ax.Axis],
        state_vectors: np.ndarray,
        last_anx_at_start=None,
        anx_times_evaluator: Union[
            Callable[[GeneralSarOrbit], np.ndarray], None, _MISSING_TYPE
        ] = MISSING,
    ):
        """
        Parameters
        ----------
        time_axis : Union[np.ndarray, are_ax.Axis]
            time axis of length N, as Axis or numpy array of PreciseDateTime objects
        state_vectors : np.ndarray
            state vectors as 1D numpy array of size 3N in the form [x0,y0,z0,x1,y1,z1,...]
        last_anx_at_start : Tuple, optional
            a tuple with the time and position as an array like, by default None
        anx_times_evaluator : Union[Callable[[GeneralSarOrbit], np.ndarray], None, _MISSING_TYPE], optional
            a function that compute ANX times from the orbit, by default MISSING

        Raises
        ------
        RuntimeError
            in case of invalid input
        ValueError
            in case of invalid input
        """
        if anx_times_evaluator is MISSING:
            anx_times_evaluator = compute_anx_times

        if isinstance(time_axis, np.ndarray):
            time_axis_start = time_axis[0]
            relative_time_axis = (time_axis - time_axis_start).astype(float)
            time_axis = are_ax.Axis(relative_time_axis, time_axis_start)

        if not isinstance(time_axis.start, PreciseDateTime):
            raise RuntimeError(
                f"Input time_axis start type: {type(time_axis.start)} != {PreciseDateTime}"
            )

        if time_axis.size != state_vectors.size / 3:
            raise RuntimeError(
                f"state vectors size incompatible with time axis size: {time_axis.size} != {state_vectors.size / 3}"
            )

        if time_axis.size < GeneralSarOrbit.get_minimum_number_of_data_points():
            raise RuntimeError(
                "Not enough state vectors provided. "
                + f"{time_axis.size} < {GeneralSarOrbit.get_minimum_number_of_data_points()}"
            )

        self._time_axis = time_axis

        if state_vectors.ndim > 1:
            raise RuntimeError(
                "input state vectors should be in the form [x0,y0,z0,x1,y1,z1,...]"
            )

        # state_vector are stored as (3, N) numpy array
        self._state_vectors = np.vstack(
            (state_vectors[::3], state_vectors[1::3], state_vectors[2::3])
        )
        self._interpolator = GeometryInterpolator(self._time_axis, self._state_vectors)

        anx_times: Optional[np.ndarray] = None
        anx_positions: Optional[npt.NDArray[np.floating]] = None

        if last_anx_at_start is not None:
            last_anx_time_at_start, last_anx_position_at_start = last_anx_at_start

            if last_anx_time_at_start is None and last_anx_position_at_start is None:
                pass
            elif last_anx_time_at_start is None or last_anx_position_at_start is None:
                raise ValueError("Both ANX time and position must be provided")
            else:
                if last_anx_time_at_start >= self._time_axis.start:
                    raise ValueError(
                        "Provided ANX time is not valid: it must be prior to time axis start"
                    )

                anx_times = np.array(last_anx_time_at_start).reshape((-1,))
                anx_positions = np.array(last_anx_position_at_start).reshape((3, -1))

        if anx_times_evaluator is not None:
            assert not isinstance(anx_times_evaluator, _MISSING_TYPE)
            evaluated_anx_times = anx_times_evaluator(self)
            evaluated_anx_positions = self.get_position(evaluated_anx_times)

            if anx_times is None and anx_positions is None:
                anx_times = evaluated_anx_times
                anx_positions = evaluated_anx_positions
            else:
                anx_times = np.concatenate((anx_times, evaluated_anx_times))
                anx_positions = np.concatenate(
                    (anx_positions, evaluated_anx_positions), axis=1
                )

        self._anx_times, self._anx_positions = anx_times, anx_positions

    def get_position(
        self, time_points, interval_indexes=None
    ) -> npt.NDArray[np.floating]:
        """Return the sensor positions at the specified time points

        Parameters
        ----------
        time_points : np.ndarray
            1D numpy array of length N of absolute time points
        interval_indexes : np.ndarray, optional
            intervals of the time axis where the given time points are expected, by default None

        Returns
        -------
        npt.NDArray[np.floating]
            (3, N) numpy array of sensor positions
        """
        return self.interpolator.eval(time_points, interval_indexes)

    def get_velocity(
        self, time_points, interval_indexes=None
    ) -> npt.NDArray[np.floating]:
        """Return the sensor velocities at the specified time points

        Velocity is evaluated using the first derivative of the interpolated position

        Parameters
        ----------
        time_points : np.ndarray
            1D numpy array of length N of absolute time points
        interval_indexes : np.ndarray, optional
            intervals of the time axis where the given time points are expected, by default None

        Returns
        -------
        npt.NDArray[np.floating]
            (3, N) numpy array of sensor velocity vectors
        """
        return self.interpolator.eval_first_derivative(time_points, interval_indexes)

    def get_acceleration(
        self, time_points, interval_indexes=None
    ) -> npt.NDArray[np.floating]:
        """Return the sensor accelerations at the specified time points

        Acceleration is evaluated using the second derivative of the interpolated position

        Parameters
        ----------
        time_points : np.ndarray
            1D numpy array of length N of absolute time points
        interval_indexes : np.ndarray, optional
            intervals of the time axis where the given time points are expected, by default None

        Returns
        -------
        npt.NDArray[np.floating]
            (3, N) numpy array of sensor acceleration vectors
        """
        return self.interpolator.eval_second_derivative(time_points, interval_indexes)

    def get_time_since_anx(
        self, time_points
    ) -> Union[
        Tuple[None, None], Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]
    ]:
        """Return the relative times from the previous ANX time

        Parameters
        ----------
        time_points : np.ndarray
            1D numpy array of length N of absolute time points

        Returns
        -------
        Union[Tuple[None, None], Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]]
            a pair of None if ANX information are missing or a tuple of two 1D numpy arrays of relative times and
            related time_nodes indices, nan value and not valid index are returned when previous time node
            is not available
        """
        if self.anx_times is None:
            return None, None

        return compute_relative_times(time_points, self.anx_times)

    def sat2earth(
        self,
        time_point,
        range_times,
        look_direction,
        geodetic_altitude=0.0,
        doppler_centroid=None,
        carrier_wavelength=None,
        orbit_tx=None,
        bistatic_delay=True,
    ):
        """Compute monostatic or bistatic sensor-to-earth projection

        Parameters
        ----------
        time_point : PreciseDateTime
            absolute time point
        range_times : npt.ArrayLike
            a vector (N, 1) of range times or a single range time, as numpy array or scalar
        look_direction : str
            either 'LEFT' or 'RIGHT'
        geodetic_altitude : float, optional
            geodetic altitude over WGS84, by default 0.0
        doppler_centroid : npt.ArrayLike, optional
            doppler centroid frequency as scalar or numpy array of size (N, 1), by default None
        carrier_wavelength : float, optional
            sensor carrier wavelength, by default None
        orbit_tx : GeneralSarOrbit, optional
            orbit of the TX sensor (for bistatic geocoding), by default None
        bistatic_delay : bool, optional
            set it to false to evaluate TX sensor position at the RX time (for bistatic case), by default True

        Returns
        -------
        np.ndarray
            (3, N) numpy array of earth positions in xyz coordinates
        """
        range_times_checked, doppler_centroid_checked = _check_sat2earth_input(
            time_point, range_times, doppler_centroid, carrier_wavelength
        )

        doppler_centroid_array = (
            np.zeros(range_times_checked.shape)
            if doppler_centroid_checked is None
            else doppler_centroid_checked
        )
        carrier_wavelength = 1.0 if carrier_wavelength is None else carrier_wavelength

        position_rx = self.get_position(time_point).squeeze()
        velocity_rx = self.get_velocity(time_point).squeeze()
        if orbit_tx is None:
            return np.atleast_2d(
                direct_geocoding_monostatic(
                    sensor_positions=position_rx,
                    sensor_velocities=velocity_rx,
                    range_times=range_times_checked,
                    geocoding_side=look_direction,
                    geodetic_altitude=geodetic_altitude,
                    frequencies_doppler_centroid=doppler_centroid_array,
                    wavelength=carrier_wavelength,
                )
            ).T

        if bistatic_delay:
            tx_time_points = [
                time_point - range_time for range_time in range_times_checked
            ]
        else:
            tx_time_points = [time_point for _ in range_times_checked]

        positions_tx = orbit_tx.get_position(tx_time_points)
        velocities_tx = orbit_tx.get_velocity(tx_time_points)

        return np.atleast_2d(
            direct_geocoding_bistatic(
                sensor_positions_rx=position_rx,
                sensor_velocities_rx=velocity_rx,
                sensor_positions_tx=positions_tx,
                sensor_velocities_tx=velocities_tx,
                range_times=range_times_checked,
                geocoding_side=look_direction,
                geodetic_altitude=geodetic_altitude,
                frequencies_doppler_centroid=doppler_centroid_array,
                wavelength=carrier_wavelength,
            )
        ).T

    def earth2sat(
        self, earth_point, doppler_centroid=None, carrier_wavelength=None, orbit_tx=None
    ) -> Tuple[List[PreciseDateTime], List[float]]:
        """Compute monostatic or bistatic earth-to-sat projection

        Parameters
        ----------
        earth_point : np.ndarray
            xyz coordinates of the point on earth as (3,) numpy array or list
        doppler_centroid : float, optional
            doppler centroid frequency, by default None
        carrier_wavelength : float, optional
            sensor carrier wavelength, by default None
        orbit_tx : _type_, optional
            orbit of the TX sensor (for bistatic inverse geocoding), by default None

        Returns
        -------
        Tuple[List[PreciseDateTime], List[float]]
            sar coordinates as two lists of azimuth and range times, respectively
        """
        if doppler_centroid is None:
            doppler_centroid = 0.0
        if carrier_wavelength is None:
            carrier_wavelength = 1.0

        earth_point = np.asarray(earth_point)
        if earth_point.shape != (3,):
            raise RuntimeError(
                f"EarthPoint has wrong shape: {earth_point.shape} != (3,)"
            )

        trajectory_rx = GSO3DCurveWrapper(self)

        if orbit_tx is None:
            init_guess = inverse_core.inverse_geocoding_monostatic_init_core(
                trajectory=trajectory_rx,
                time_axis=self.time_axis_array,
                ground_points=earth_point,
                frequencies_doppler_centroid=doppler_centroid,
                wavelength=carrier_wavelength,
            )
            # keeping only first solution for each point
            az_initial_time_guesses = np.array([g[0] for g in init_guess])

            if az_initial_time_guesses.size == 1 and earth_point.ndim == 1:
                az_initial_time_guesses = az_initial_time_guesses[0]
            times = inverse_core.inverse_geocoding_monostatic_core(
                trajectory=trajectory_rx,
                initial_guesses=az_initial_time_guesses,
                ground_points=earth_point,
                frequencies_doppler_centroid=doppler_centroid,
                wavelength=carrier_wavelength,
            )

        else:
            trajectory_tx = GSO3DCurveWrapper(orbit_tx)
            init_guess = inverse_core.inverse_geocoding_bistatic_init_core(
                trajectory_rx=trajectory_rx,
                trajectory_tx=trajectory_tx,
                time_axis_rx=self.time_axis_array,
                time_axis_tx=orbit_tx.time_axis_array,
                ground_points=earth_point,
                frequencies_doppler_centroid=doppler_centroid,
                wavelength=carrier_wavelength,
            )
            times = inverse_core.inverse_geocoding_bistatic_core(
                trajectory_rx=trajectory_rx,
                trajectory_tx=trajectory_tx,
                initial_guesses=init_guess,
                ground_points=earth_point,
                frequencies_doppler_centroid=doppler_centroid,
                wavelength=carrier_wavelength,
            )

        az, rg = times
        if isinstance(az, PreciseDateTime):
            assert isinstance(rg, float)
            return ([az], [rg])

        assert isinstance(rg, np.ndarray)
        return (az.tolist(), rg.tolist())

    def evaluate_doppler_equation(
        self, earth_point: np.ndarray, doppler_centroid, carrier_wavelength
    ):
        """Evaluate the doppler equation over the orbit time axis

        .. math::

            \\frac{2}{\\lambda} \\frac{(P_{sat}(t) - P_0) \\cdot V_{sat}(t)}{||P_{sat}(t) - P_0||} - f_{DC}

        Parameters
        ----------
        earth_point : np.ndarray
            xyz coordinates of the point on earth as (3,) numpy array
        doppler_centroid : float
            doppler centroid frequency
        carrier_wavelength : float
            sensor carrier wavelength

        Returns
        -------
        np.ndarray
            resulting doppler equation values in a numpy array of size N
        """
        return doppler_equation(
            earth_point,
            self._state_vectors,
            self.get_velocity(self._time_axis.get_array()),
            doppler_centroid,
            carrier_wavelength,
        )

    def __repr__(self):
        axis_str = str(self._time_axis)
        state_vec_str = str(self._state_vectors)

        axis_portion = (
            "Orbit defined on azimuth axis: " + os.linesep + axis_str + os.linesep
        )
        state_vectors_portion = (
            "State vectors: " + os.linesep + state_vec_str + os.linesep
        )
        return axis_portion + state_vectors_portion

    def get_interpolated_time_axis(self, interpolation_positions):
        """Return the interpolated time axis on the given interpolation positions

        Parameters
        ----------
        interpolation_positions : float
            fracional indexes

        Returns
        -------
        PreciseDateTime
            time point

        See also
        --------
        arepytools.math.axis.Axis.interpolate : time axis interpolation method
        """
        return self._time_axis.interpolate(interpolation_positions)


def _identify_anx_time_intervals(reference_time_axis, position_evaluator):
    number_of_intervals = reference_time_axis.size - 1
    evaluated_positions = position_evaluator(reference_time_axis)

    anx_intervals = []

    for interval_index in range(number_of_intervals):
        interval_begin_index, interval_end_index = interval_index, interval_index + 2

        z_start, z_stop = evaluated_positions[
            2, interval_begin_index:interval_end_index
        ]
        is_anx_interval = z_start <= 0 < z_stop

        if is_anx_interval:
            t_start, t_stop = reference_time_axis[
                interval_begin_index:interval_end_index
            ]
            anx_intervals.append((t_start, t_stop))

    return anx_intervals


def compute_anx_times(
    orbit: GeneralSarOrbit,
    *,
    max_abs_z_error=1e-3,
    max_abs_time_error=1e-6,
    max_search_iterations=100,
) -> np.ndarray:
    """Compute ANX times of the specified orbit

    Parameters
    ----------
    orbit : GeneralSarOrbit
        orbit
    max_abs_z_error : float, optional
        maximum absolute error on ANX z coordinate, by default 1e-3
    max_abs_time_error : float, optional
        maximum absolute error on ANX time, by default 1e-6
    max_search_iterations : int, optional
        maximum number of search iterations per ANX, by default 100

    Returns
    -------
    np.ndarray
        (N,) array of ANX times

    Raises
    ------
    ValueError
        in case of invalid input
    """
    if max_abs_z_error <= 0:
        raise ValueError("Invalid maximum absolute ANX z-coordinate error value")

    if max_abs_time_error <= 0:
        raise ValueError("Invalid maximum absolute ANX time error value")

    if max_search_iterations <= 0:
        raise ValueError("Invalid maximum number of search iterations per ANX")

    anx_time_intervals = _identify_anx_time_intervals(
        orbit.time_axis_array, orbit.get_position
    )

    anx_times = np.empty((len(anx_time_intervals),), dtype=orbit.time_axis_array.dtype)

    def get_z_coordinate(time, origin):
        return orbit.get_position(origin + time)[2]

    for interval_index, anx_time_interval in enumerate(anx_time_intervals):
        reference_time = anx_time_interval[0]

        central_time = (
            reference_time + (anx_time_interval[1] - anx_time_interval[0]) / 2.0
        )
        velocity_z = orbit.get_velocity(central_time)[2]
        xtol = min(max_abs_time_error, max_abs_z_error / abs(velocity_z))

        get_z_coordinate = functools.partial(get_z_coordinate, origin=reference_time)

        anx_time_interval_rel = [t - reference_time for t in anx_time_interval]
        anx_time_rel = scipy.optimize.bisect(
            get_z_coordinate,
            *anx_time_interval_rel,
            xtol=xtol,
            maxiter=max_search_iterations,
        )

        anx_times[interval_index] = reference_time + anx_time_rel

    return anx_times


def compute_number_of_anx(orbit: GeneralSarOrbit) -> int:
    """Compute the number of ANX of the specified orbit

    Parameters
    ----------
    orbit : GeneralSarOrbit
        sensor orbit

    Returns
    -------
    int
        number of ANX in the orbit
    """
    anx_time_intervals = _identify_anx_time_intervals(
        orbit.time_axis_array, orbit.get_position
    )

    return len(anx_time_intervals)


def compute_relative_times(
    time_points, time_nodes
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
    """Return the relative time since the greatest node less or equal to absolute time

    Parameters
    ----------
    time_points : np.ndarray
        1D numpy array of length N of absolute time points
    time_nodes : np.ndarray
        1D numpy array of sorted absolute times

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]
        a tuple of two 1D numpy arrays of relative times and related time_nodes indices,
        nan value and not valid index are returned when previous time node is not available
    """
    relative_times = np.empty_like(time_points, dtype=float)
    node_indices = np.empty_like(time_points, dtype=int)

    for time_index, time_value in np.ndenumerate(time_points):
        lower_time_nodes = time_nodes[time_nodes <= time_value]
        if len(lower_time_nodes) > 0:
            relative_times[time_index] = time_value - lower_time_nodes[-1]
            node_indices[time_index] = len(lower_time_nodes) - 1
        else:
            relative_times[time_index] = np.nan
            node_indices[time_index] = len(time_nodes)

    return relative_times, node_indices


def create_general_sar_orbit(
    state_vectors: StateVectors, ignore_anx_after_orbit_start=False
) -> GeneralSarOrbit:
    """Create general sar orbit object from state vectors metadata

    Parameters
    ----------
    state_vectors : StateVectors
        state vectors as a StateVectors metadata object
    ignore_anx_after_orbit_start : bool, default False
        if true, the ANX time in state_vectors is ignored in case it is not immediately
        antecedent to the orbit start

    Returns
    -------
    GeneralSarOrbit
        the new GeneralSarOrbit object
    """
    time_axis = are_ax.RegularAxis(
        (0, state_vectors.time_step, state_vectors.number_of_state_vectors),
        state_vectors.reference_time,
    )

    if (
        ignore_anx_after_orbit_start
        and state_vectors.anx_time is not None
        and state_vectors.anx_time >= time_axis.start
    ):
        last_anx_at_start = None
    else:
        last_anx_at_start = (state_vectors.anx_time, state_vectors.anx_position)

    return GeneralSarOrbit(
        time_axis,
        state_vectors.position_vector.reshape((state_vectors.position_vector.size,)),
        last_anx_at_start,
    )


def compute_look_angles_from_orbit(
    orbit: GeneralSarOrbit,
    azimuth_time: PreciseDateTime,
    range_times: npt.ArrayLike,
    look_direction,
    *,
    altitude_over_wgs84: float = 0.0,
    doppler_centroid: Optional[npt.ArrayLike] = None,
    carrier_wavelength: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Perform direct geocoding and then compute look angles

    Parameters
    ----------
    orbit : GeneralSarOrbit
        orbit
    azimuth_time : PreciseDateTime
        absolute time point
    range_times : npt.ArrayLike
        scalar or (N,) array like
    look_direction : str
        either 'LEFT' or 'RIGHT'
    altitude_over_wgs84 : float, optional
        altitude over WGS84, by default 0.0
    doppler_centroid : Optional[npt.ArrayLike], optional
        doppler centroid frequency as scalar or (N,), by default None
    carrier_wavelength : Optional[float], optional
        sensor carrier wavelength, by default None

    Returns
    -------
    Union[float, np.ndarray]
        scalar or (N,) look angles in radians
    """

    points = orbit.sat2earth(
        azimuth_time,
        range_times,
        look_direction,
        altitude_over_wgs84,
        doppler_centroid,
        carrier_wavelength,
    ).T
    points = points.reshape(np.shape(range_times) + (3,))

    sensor_position = orbit.get_position(azimuth_time).squeeze()

    sensor_position_ground = conversions.xyz2llh(sensor_position)
    sensor_position_ground[2] = 0.0
    sensor_position_ground = conversions.llh2xyz(sensor_position_ground).squeeze()

    nadir = sensor_position_ground - sensor_position

    return compute_look_angles(sensor_position.T, nadir.T, points)


def compute_incidence_angles_from_orbit(
    orbit: GeneralSarOrbit,
    azimuth_time: PreciseDateTime,
    range_times: npt.ArrayLike,
    look_direction,
    *,
    altitude_over_wgs84: float = 0.0,
    doppler_centroid: Optional[npt.ArrayLike] = None,
    carrier_wavelength: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Perform direct geocoding and then compute incidence angles

    Parameters
    ----------
    orbit : GeneralSarOrbit
        orbit
    azimuth_time : PreciseDateTime
        absolute time point
    range_times : npt.ArrayLike
        scalar or (N,) array like
    look_direction : str
        either 'LEFT' or 'RIGHT'
    altitude_over_wgs84 : float, optional
        altitude over WGS84, by default 0.0
    doppler_centroid : Optional[npt.ArrayLike], optional
        doppler centroid frequency as scalar or (N,), by default None
    carrier_wavelength : Optional[float], optional
        sensor carrier wavelength, by default None

    Returns
    -------
    Union[float, np.ndarray]
        scalar or (N,) incidence angles in radians
    """
    points = orbit.sat2earth(
        azimuth_time,
        range_times,
        look_direction,
        altitude_over_wgs84,
        doppler_centroid,
        carrier_wavelength,
    ).T
    points = points.reshape(np.shape(range_times) + (3,))

    sensor_position = orbit.get_position(azimuth_time).T.squeeze()

    return compute_incidence_angles(sensor_position, points)


def compute_ground_velocity(
    orbit: GeneralSarOrbit,
    time_point: PreciseDateTime,
    look_angles: npt.ArrayLike,
    *,
    reference_frame: ReferenceFrameLike = ReferenceFrame.zero_doppler,
    altitude_over_wgs84: float = 0,
    averaging_interval_relative_origin: float = 0,
    averaging_interval_duration: float = 1.0,
    averaging_interval_num_points: int = 11,
) -> Union[float, np.ndarray]:
    """Compute numerically the ground velocity at given look angles.

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
    look_angles = np.asarray(look_angles)

    averaging_time_axis = (
        np.linspace(
            averaging_interval_relative_origin,
            averaging_interval_duration,
            averaging_interval_num_points,
        )
        + time_point
    )

    sensor_positions = orbit.get_position(averaging_time_axis).T
    sensor_velocities = orbit.get_velocity(averaging_time_axis).T

    points = np.empty(look_angles.shape + (averaging_time_axis.size, 3))
    for look_angle, point in zip(
        look_angles.reshape((-1,)),
        points.reshape((-1, averaging_time_axis.size, 3)),
    ):
        point[:, :] = direct_geocoding_with_look_angles(
            sensor_positions,
            sensor_velocities,
            reference_frame,
            look_angle,
            altitude_over_wgs84=altitude_over_wgs84,
        )

    points_difference = np.diff(points, axis=-2)
    distances = np.linalg.norm(points_difference, axis=-1)
    return np.sum(distances, axis=-1) / averaging_interval_duration


def _check_sat2earth_input(
    azimuth_time, range_times, frequency_doppler_centroid, wavelength
):
    if not isinstance(azimuth_time, PreciseDateTime):
        raise RuntimeError("Azimuth should be a single absolute time")

    if isinstance(range_times, (list, np.ndarray)):
        range_times = np.asarray(range_times)
    else:
        range_times = np.full((1,), range_times)

    if (frequency_doppler_centroid is not None and wavelength is None) or (
        wavelength is not None and frequency_doppler_centroid is None
    ):
        raise RuntimeError(
            "Frequency doppler centroid and wavelength should be both specified"
        )

    if frequency_doppler_centroid is not None:
        if isinstance(frequency_doppler_centroid, (list, np.ndarray)):
            frequency_doppler_centroid = np.asarray(frequency_doppler_centroid)
        else:
            frequency_doppler_centroid = np.full((1,), frequency_doppler_centroid)

    if frequency_doppler_centroid is not None and (
        frequency_doppler_centroid.ndim != 1
        or frequency_doppler_centroid.shape != range_times.shape
    ):
        if frequency_doppler_centroid.size == 1:
            frequency_doppler_centroid = np.full(
                range_times.shape, frequency_doppler_centroid[0]
            )
        else:
            raise RuntimeError(
                "Frequency doppler centroid vector should have the same shape of the range times vector: "
                + f"{frequency_doppler_centroid.shape} != {range_times.shape}"
            )

    return range_times, frequency_doppler_centroid
