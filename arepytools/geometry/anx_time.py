# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
ANX Time module
---------------
"""

from __future__ import annotations

import functools

import numpy as np
import scipy.optimize

from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve


def compute_anx_times_core(
    trajectory: TwiceDifferentiable3DCurve,
    time_sampling_step_s: float = 1,
    max_abs_z_error: float = 1e-3,
    max_abs_time_error: float = 1e-6,
    max_search_iterations: float = 100,
) -> np.ndarray:
    """Compute ANX times of the specified trajectory.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor's trajectory from which compute the ANX times
    time_sampling_step_s : float, optional
        sampling step in seconds for the trajectory time axis, by default 1
    max_abs_z_error : float, optional
        maximum absolute error on ANX z coordinate, by default 1e-3
    max_abs_time_error : float, optional
        maximum absolute error on ANX time, by default 1e-6
    max_search_iterations : float, optional
        maximum number of search iterations per ANX, by default 100

    Returns
    -------
    np.ndarray
        (N,) array of ANX times
    """

    time_axis_origin = trajectory.domain[0]
    time_axis_rel = np.arange(
        0, trajectory.domain[1] - trajectory.domain[0], time_sampling_step_s
    )
    evaluated_positions = trajectory.evaluate(time_axis_rel + time_axis_origin)

    anx_time_rel_intervals = _find_anx_time_intervals(
        time_axis_rel=time_axis_rel, positions=evaluated_positions
    )

    def get_z_coordinate_bisecting_func(time, origin):
        return trajectory.evaluate(origin + time)[-1]

    anx_times = []
    for interval in anx_time_rel_intervals:

        central_time = interval[0] + (interval[1] - interval[0]) / 2.0
        velocity_z = trajectory.evaluate_first_derivatives(
            central_time + time_axis_origin
        )[-1]
        x_tol = min(max_abs_time_error, max_abs_z_error / abs(velocity_z))

        get_z_coordinate_bisecting_func = functools.partial(
            get_z_coordinate_bisecting_func, origin=interval[0] + time_axis_origin
        )

        relative_interval = [t - interval[0] for t in interval]
        anx_time_rel = scipy.optimize.bisect(
            get_z_coordinate_bisecting_func,
            *relative_interval,
            xtol=x_tol,
            maxiter=max_search_iterations,
        )

        anx_times.append(interval[0] + anx_time_rel)

    return np.array(anx_times) + time_axis_origin


def compute_relative_times(
    time_points: np.ndarray, anx_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the relative time since the greatest node less or equal to absolute time.

    Parameters
    ----------
    time_points : np.ndarray
        absolute time points, as an array of (N,)
    anx_times : np.ndarray
        sorted absolute anx times, as an array of (N,)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        a tuple of two 1D numpy arrays of relative times and related time_nodes indices,
        nan value and not valid index are returned when previous time node is not available
    """
    relative_times = np.array(
        [
            (
                t - anx_times[anx_times <= t][-1]
                if len(anx_times[anx_times <= t]) > 0
                else np.nan
            )
            for t in time_points
        ]
    )
    anx_indices = np.array(
        [
            (
                len(anx_times[anx_times <= t]) - 1
                if len(anx_times[anx_times <= t]) > 0
                else len(anx_times)
            )
            for t in time_points
        ]
    ).astype(int)

    return relative_times, anx_indices


def _find_anx_time_intervals(
    time_axis_rel: np.ndarray, positions: np.ndarray
) -> list[tuple[float, float]]:
    """Finding ANX time intervals for the input relative time axis and the corresponding sensor's positions.

    Parameters
    ----------
    time_axis_rel : np.ndarray
        relative time axis, as a (N,) array of floats
    positions : np.ndarray
        sensor's positions corresponding to the input times, as a (N, 3) array of floats

    Returns
    -------
    list[tuple[float, float]]
        list of ANX time intervals as (starting time, ending time) of the interval crossing ANX
    """
    anx_intervals = []
    for interval_index in range(time_axis_rel.size - 1):
        interval_begin_index, interval_end_index = interval_index, interval_index + 2

        z_start, z_stop = positions[interval_begin_index:interval_end_index, -1]
        is_anx_interval = z_start <= 0 < z_stop

        if is_anx_interval:
            t_start, t_stop = time_axis_rel[interval_begin_index:interval_end_index]
            anx_intervals.append((t_start, t_stop))

    return anx_intervals
