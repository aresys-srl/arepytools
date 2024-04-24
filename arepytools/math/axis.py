# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Axis module
-----------
"""

import os
import typing
from typing import Optional

import numpy as np
import scipy.interpolate

from arepytools.timing.precisedatetime import PreciseDateTime

# Types
RealNumber = typing.Union[int, float]
AxisStartType = typing.Union[RealNumber, PreciseDateTime]
RegularAxisTuple = typing.Tuple[RealNumber, RealNumber, int]
GeneralAxisVector = np.ndarray


class Axis:
    """Generic axis"""

    def __init__(self, relative_axis: GeneralAxisVector, origin: AxisStartType = 0):
        """Initialize an axis as origin + relative_axis
        The new axis starts at origin + relative_axis[0]

        Parameters
        ----------
        relative_axis : GeneralAxisVector
            a monotone array either increasing or decreasing
        origin : AxisStartType, optional
            axis origin, by default 0

        Raises
        ------
        ValueError
            in case of invalid input parameters
        """
        valid, error_message = _validate_general_axis(relative_axis)
        if not valid:
            raise ValueError(error_message)

        self._relative_axis = relative_axis

        if not isinstance(origin, (PreciseDateTime, int, float)):
            raise ValueError(
                "origin should be either a PreciseDateTime or a real number"
            )

        self._origin = origin

        self._increasing = _is_increasing(self._relative_axis)

    @property
    def start(self) -> AxisStartType:
        """Axis start point (origin + relative_axis[0])

        See Also
        --------
        origin : the origin parameter
        """
        return self._origin + self._relative_axis[0]

    @property
    def origin(self) -> AxisStartType:
        """Axis origin

        See Also
        --------
        start : the actual start point of the axis
        """
        return self._origin

    @property
    def increasing(self) -> bool:
        """whether the axis is monotone increasing"""
        return self._increasing

    @property
    def decreasing(self) -> bool:
        """whether the axis is monotone decreasing"""
        return not self.increasing

    @property
    def mean_step(self) -> RealNumber:
        """the average step"""
        return np.diff(self._relative_axis).mean()

    @property
    def size(self) -> int:
        """number of points in the axis"""
        return self._relative_axis.size

    @property
    def length(self) -> RealNumber:
        """length of the internal part of the axis (N - 1) * step"""
        return abs(self._relative_axis[-1] - self._relative_axis[0])

    def get_array(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        """Get the a portion of the axis

        The entire axis is returned by default

        Parameters
        ----------
        start : int, optional
            start index (included), by default 0
        stop : int, optional
            stop index (excluded), by default None

        Returns
        -------
        np.ndarray
            portion of the axis
        """
        return self.get_relative_array(start, stop) + self._origin

    def get_relative_array(self, start=0, stop=None) -> np.ndarray:
        """Get a relative portion of the axis

        Parameters
        ----------
        start : int, optional
            start index (included), by default 0
        stop : int, optional
            stop index (excluded), by default None

        Returns
        -------
        np.ndarray
            portion of the axis, relative to the origin
        """
        stop = self.size if stop is None else stop
        _check_range(start, stop, self.size)
        return self._relative_axis[start:stop]

    def get_interval_id(self, values) -> np.ndarray:
        """For each value, find the containing axis interval

        If values lie outside the axis the closest intervals are returned.

        Parameters
        ----------
        values : np.ndarray
            values to locate on the axis

        Returns
        -------
        np.ndarray
            the positions of the values in the axis

        See also
        --------
        get_interval_id_from_relative : for relative values
        """
        return _get_interval_id_not_regular_real_axis(
            self._relative_axis, self.increasing, values - self._origin
        )

    def get_interval_id_from_relative(self, values) -> np.ndarray:
        """For each relative value, find the containing axis interval

        If values lie outside the axis the closest intervals are returned.
        Values are intended relative to the origin

        Parameters
        ----------
        values : np.ndarray
            values to locate on the axis

        Returns
        -------
        np.ndarray
            the positions of the values in the axis

        See also
        --------
        origin : origin property
        get_interval_id : for absolute values
        """
        return self.get_interval_id(values + self._origin)

    def interpolate(self, position):
        """Interpolate axis values at the given position

        It coincides with accessing the array at the given position when the position is an integer,
        otherwise it interpolates the closest values

        Parameters
        ----------
        position : float
            fractional index

        Returns
        -------
        AxisStartType
            the value of the axis at the fraction index
        """
        return (
            scipy.interpolate.interp1d(range(self.size), self._relative_axis)(position)
            + self._origin
        )

    def __repr__(self):
        """string representation of the object"""
        axis_repr = (
            "{classname} -- direction: {direction}:"
            + "{newline} start: {start}, end: {end}"
            + "{newline} size: {size}, length: {length}"
        )
        return axis_repr.format(
            start=self.start,
            size=self.size,
            end=self.start + self.length if self.increasing else -self.length,
            newline=os.linesep + "--",
            classname=self.__class__.__name__,
            direction="increasing" if self.increasing else "decreasing",
            length=self.length,
        )


class RegularAxis(Axis):
    """Regular axis class"""

    def __init__(self, relative_axis: RegularAxisTuple, origin: AxisStartType = 0):
        """

        A RegularAxis is an Axis where the relative_axis is given in the form:

        .. math:: origin + start + k \\cdot step, \\,k=0\\dots size-1

        Parameters
        ----------
        relative_axis : RegularAxisTuple
            a tuple (start, step, size)
        origin : AxisStartType, optional
            axis origin, by default 0

        Raises
        ------
        ValueError
            in case of invalid input
        """
        # input validation
        valid, error_message = _validate_uniform_axis(relative_axis)
        if not valid:
            raise ValueError(error_message)

        # relative axis initialization
        start, step, size = relative_axis
        super().__init__(np.asarray([start + step * k for k in range(size)]), origin)

    @property
    def step(self) -> RealNumber:
        """Axis step"""
        return self.mean_step

    def interpolate(self, position):
        """Interpolate axis values at the given position

        .. math:: start + position \\cdot step

        Parameters
        ----------
        position : float
            fractional index

        Returns
        -------
        AxisStartType
            the value of the axis at the fraction index
        """
        return position * self.step + self.start

    def get_interval_id(self, values) -> np.ndarray:
        """For each value, find the containing axis interval

        If values lie outside the axis the closest intervals are returned.

        Parameters
        ----------
        values : np.ndarray
            values to locate on the axis

        Returns
        -------
        np.ndarray
            the positions of the values in the axis

        See also
        --------
        Axis.get_interval_id_from_relative : for relative values
        """
        return _get_interval_id_regular_real_axis(
            self._relative_axis[0], self.step, self.size, values - self._origin
        )


def _check_range(start, stop, size):
    if stop > size:
        raise ValueError(f"Stop ({stop}) should be smaller equal than size: ({size})")
    if start < 0:
        raise ValueError(f"Start ({start}) should be positive")


def _is_increasing(axis):
    steps = np.diff(axis)
    if (steps > 0).all():
        return True

    if (steps < 0).all():
        return False

    raise RuntimeError("Expecting monotone axis")


def _get_interval_id_regular_real_axis(
    start: RealNumber, step: RealNumber, size: int, values
) -> np.ndarray:
    """The interval id is intended as the interval starting at the returned position, containing the point.
    If the point is exactly an edge of the interval, the interval starting at the point should be returned.

    Parameters
    ----------
    start : RealNumber
        the start of the axis
    step : RealNumber
         the step of the axis
    size : int
        the size of the axis
    values : _type_
        the values to be located in the array

    Returns
    -------
    np.ndarray
        indexes
    """

    def to_int_and_clip(value):
        return np.clip(int(np.floor(value)), 0, size - 1)

    val = (values - start) / step

    if isinstance(val, np.ndarray):
        return np.array([to_int_and_clip(v) for v in val])

    return np.array([to_int_and_clip(val)])


def _get_interval_id_not_regular_real_axis(
    array: np.ndarray, increasing: bool, values: np.ndarray
) -> np.ndarray:
    """The interval id is intended as the interval starting at the returned position, containing the point.
    If the point is exactly an edge of the interval, the interval starting at the point should be returned.

    Parameters
    ----------
    array : np.ndarray
        the axis
    increasing : bool
        wether the axis is increasing or not
    values : np.ndarray
        the values to be located in the array

    Returns
    -------
    np.ndarray
        interval indexes
    """
    if isinstance(values, (list, np.ndarray)):
        values = np.asarray(values)
    else:
        values = np.full((1,), values)

    out = np.empty((values.size,))
    for k, v in enumerate(values):
        closest_pos = np.argmin(np.abs((array - v)))
        if increasing:
            if array[closest_pos] > v:
                out[k] = closest_pos - 1
            else:
                out[k] = closest_pos
        else:
            if array[closest_pos] < v:
                out[k] = closest_pos - 1
            else:
                out[k] = closest_pos
    return np.clip(out.astype(int), 0, array.size - 1)


def _validate_uniform_axis(uniform_axis: tuple) -> typing.Tuple[bool, str]:
    if not isinstance(uniform_axis, tuple):
        return False, "relative axis should be a tuple"

    if not len(uniform_axis) == 3:  # start, step and size
        return (
            False,
            "relative axis should be a tuple of size three (start, step, size)",
        )

    if not isinstance(uniform_axis[0], (float, int)):  # start
        return (
            False,
            f"start of relative axis {type(uniform_axis[0])} should be a real number",
        )

    if not isinstance(uniform_axis[1], (float, int)):  # step
        return (
            False,
            f"Step of relative axis {type(uniform_axis[1])} should be a real number",
        )

    if not isinstance(uniform_axis[2], int):  # step
        return (
            False,
            f"Type of relative axis {type(uniform_axis[2])} should be an integer",
        )

    if uniform_axis[2] < 1:  # step
        return False, f"Size of relative axis {uniform_axis[2]} should be an integer"

    return True, ""


def _validate_general_axis(general_axis: np.ndarray) -> typing.Tuple[bool, str]:
    if not isinstance(general_axis, np.ndarray):
        return False, "relative axis should be a numpy.ndarray"

    if general_axis.dtype not in (int, float):
        return (
            False,
            f"relative axis dtype {general_axis.dtype} should be a numpy.ndarray of real numbers",
        )

    if not general_axis.size == general_axis.shape[0]:
        return (
            False,
            f"relative axis should be a vector not a matrix {general_axis.size} != {general_axis.shape[0]}",
        )

    return True, ""
