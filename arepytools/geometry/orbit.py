# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Interpolated Orbit module
-------------------------
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline


class ExtrapolationNotAllowed(ValueError):
    """Orbit trajectory extrapolation outside of time boundaries is not allowed"""


class Orbit:
    """Orbit object based on a Cubic Spline interpolator keeping continuity up to second derivatives"""

    def __init__(
        self, times: np.ndarray, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        """Orbit object creation depending on positions, velocities and time axis.
        Time axis can be specified as relative or absolute (actual dates), while positions and velocities must be
        specified as (N, 3) arrays of floats.

        This object is based on a scipy CubicSpline interpolator.
        **Extrapolation** outside time validity boundaries **is not allowed**.

        Parameters
        ----------
        times : np.ndarray
            time axis as numpy array of shape (N,) or (N, 1), it can be relative or absolute
        positions : np.ndarray
            positions as numpy array of shape (N, 3), with coordinates being x, y, z
        velocities : np.ndarray
            velocities as numpy array of shape (N, 3), with coordinates being x, y, z
        """
        assert positions.shape[1] == 3 and positions.ndim == 2
        assert velocities.shape[1] == 3 and velocities.ndim == 2
        self._positions = positions
        self._velocities = velocities
        self._times = times
        self._time_origin = times.squeeze()[0]
        self._last_time = times.squeeze()[-1]
        self._time_relative = times.squeeze() - self._time_origin
        self._domain = (self._time_origin, self._last_time)
        self._interpolator = self._create_interpolator()

    @property
    def positions(self) -> np.ndarray:
        """Accessing orbit positions vector"""
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """Accessing orbit velocities vector"""
        return self._velocities

    @property
    def times(self) -> np.ndarray:
        """Accessing orbit times vector"""
        return self._times

    @property
    def domain(self) -> np.ndarray:
        """Orbit time domain"""
        return self._domain

    def _create_interpolator(self) -> CubicSpline:
        """Generating the Cubic Spline interpolator from given inputs.

        Returns
        -------
        CubicSpline
            CubicSpline scipy interpolator object
        """
        return CubicSpline(
            x=self._time_relative,
            y=self._positions,
            bc_type=((1, self._velocities[0, :]), (1, self._velocities[-1, :])),
            extrapolate=False,
        )

    def _check_time_validity(self, times: npt.ArrayLike) -> None:
        """Check input times validity with respect to the time validity boundaries. Extrapolation is not allowed.

        Parameters
        ----------
        times : Union[float, npt.ArrayLike]
            input times at which interpolate the trajectory

        Raises
        ------
        ExtrapolationNotAllowed
            if one or more of the input times is not inside the time boundaries of trajectory definition
        """
        if np.any(times < self._time_origin) or np.any(times > self._last_time):
            raise ExtrapolationNotAllowed(
                "One (or more) of the input times is outside of trajectory time boundaries"
            )

    def evaluate(self, times: npt.ArrayLike) -> np.ndarray:
        """Evaluate x, y, z interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        times : npt.ArrayLike
            time coordinates compatible with the time type used for building the Orbit interpolator

        Returns
        -------
        np.ndarray
            interpolated values for x, y, z at given times
        """
        self._check_time_validity(times)
        relative_times = times - self._time_origin
        return self._interpolator(relative_times, 0, extrapolate=False)

    def evaluate_first_derivatives(self, times: npt.ArrayLike) -> np.ndarray:
        """Evaluate x, y, z first derivatives (vx, vy, vz) interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        times : npt.ArrayLike
            time coordinates compatible with the time type used for building the Orbit interpolator

        Returns
        -------
        np.ndarray
            interpolated first derivatives values for x, y, z at given times
        """
        self._check_time_validity(times)
        relative_times = times - self._time_origin
        return self._interpolator(relative_times, 1, extrapolate=False)

    def evaluate_second_derivatives(self, times: npt.ArrayLike) -> np.ndarray:
        """Evaluate x, y, z second derivatives (ax, ay, az) interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        times : npt.ArrayLike
            time coordinates compatible with the time type used for building the Orbit interpolator

        Returns
        -------
        np.ndarray
            interpolated second derivatives values for x, y, z at given times
        """
        self._check_time_validity(times)
        relative_times = times - self._time_origin
        return self._interpolator(relative_times, 2, extrapolate=False)
