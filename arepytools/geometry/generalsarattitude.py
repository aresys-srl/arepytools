# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
General SAR attitude module
---------------------------
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np
import numpy.typing as npt

from arepytools.geometry import curve
from arepytools.geometry._interpolator import GeometryInterpolator
from arepytools.geometry.attitude_utils import (
    compute_antenna_reference_frame_from_euler_angles,
)
from arepytools.geometry.direct_geocoding import direct_geocoding_with_looking_direction
from arepytools.geometry.generalsarorbit import (
    GeneralSarOrbit,
    create_general_sar_orbit,
)
from arepytools.geometry.reference_frames import (
    ReferenceFrame,
    compute_sensor_local_axis,
)
from arepytools.geometry.rotation import RotationOrder, compute_rotation
from arepytools.io.metadata import AttitudeInfo, StateVectors
from arepytools.math import axis as are_ax
from arepytools.timing.precisedatetime import PreciseDateTime


def create_attitude_boresight_normal_curve_wrapper(
    attitude: GeneralSarAttitude,
) -> curve.Generic3DCurve:
    """Creating a Generic3DCurve wrapper for attitude boresight normal using splines interpolators.

    Parameters
    ----------
    attitude : GeneralSarAttitude
        attitude object

    Returns
    -------
    curve.Generic3DCurve
        generic 3D curve wrapper for the boresight normal component of the attitude
    """
    arf_vals = attitude.get_arf(attitude.time_axis_array)
    t_start = attitude.time_axis_array[0]
    t_end = attitude.time_axis_array[-1]
    relative_axis = attitude.time_axis_array - t_start

    # creating splines for each component of the boresight normal vector
    x_spline = curve.SplineWrapper(axis=relative_axis, values=arf_vals[:, 0, 0])
    y_spline = curve.SplineWrapper(axis=relative_axis, values=arf_vals[:, 1, 0])
    z_spline = curve.SplineWrapper(axis=relative_axis, values=arf_vals[:, 2, 0])

    return curve.Generic3DCurve(
        x_spline, y_spline, z_spline, t_start, time_boundaries=(t_start, t_end)
    )


class GeneralSarAttitude:
    """General sar attitude"""

    _MINIMUM_NUMBER_OF_DATA_POINTS = GeometryInterpolator.get_min_num_of_data_points()

    _ANGLE_INDEX = {"Y": 0, "P": 1, "R": 2}

    @classmethod
    def get_minimum_number_of_data_points(cls) -> int:
        """Return the required minimum number of data points

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
            "Time step is not available for attitudes constructed with non-regular time axis"
        )

    @property
    def n(self) -> int:
        """Number of attitude data points"""
        return self._time_axis.size

    @property
    def ypr_angles(self) -> npt.NDArray[np.floating]:
        """YPR angles as a 3xN numpy array"""
        return self._ypr_angles

    @property
    def rotation_order(self) -> RotationOrder:
        """Yaw/Pitch/Roll rotation order"""
        return self._rotation_order

    @property
    def reference_frame(self) -> ReferenceFrame:
        """Reference frame"""
        return self._reference_frame

    @property
    def time_axis_array(self) -> np.ndarray:
        """Time axis as array of time points"""
        return self._time_axis.get_array()

    @property
    def interpolator(self):
        """Geometry interpolator object"""
        return self._interpolator

    def __init__(
        self,
        orbit: GeneralSarOrbit,
        time_axis: Union[np.ndarray, are_ax.Axis],
        ypr_angles: np.ndarray,
        rotation_order: str,
        reference_frame: str,
    ):
        """
        Parameters
        ----------
        orbit : GeneralSarOrbit
            sensor orbit
        time_axis : Union[np.ndarray, are_ax.Axis]
            time axis of length N, as Axis or numpy array of PreciseDateTime objects
        ypr_angles : np.ndarray
            yaw, pitch and roll angles as (3, N) numpy array
        rotation_order : str
            rotation order string
        reference_frame : str
            reference frame string
        """
        if isinstance(time_axis, np.ndarray):
            time_axis_start = time_axis[0]
            relative_time_axis = (time_axis - time_axis_start).astype(float)
            time_axis = are_ax.Axis(relative_time_axis, time_axis_start)

        if not isinstance(time_axis.start, PreciseDateTime):
            raise RuntimeError(
                f"Input time_axis start type: {type(time_axis.start)} != {PreciseDateTime}"
            )

        if np.shape(ypr_angles) != (3, time_axis.size):
            raise RuntimeError(
                f"Invalid ypr angles shape: {np.shape(ypr_angles)} "
                + f"it should be compatible with the number of time axis {time_axis.size}"
            )

        if (
            np.shape(ypr_angles)[1]
            < GeneralSarAttitude.get_minimum_number_of_data_points()
        ):
            raise RuntimeError(
                "Not enough attitude records provided: "
                + f"{np.shape(ypr_angles)[1]} < {GeneralSarAttitude.get_minimum_number_of_data_points()}"
            )

        self._time_axis = time_axis
        self._orbit = orbit
        self._rotation_order = RotationOrder(rotation_order.upper())
        self._reference_frame = ReferenceFrame(reference_frame.upper())
        self._ypr_angles = np.array(ypr_angles)

        self._interpolator = GeometryInterpolator(self._time_axis, self._ypr_angles)

    def _interpolate_angles(self, time_points, angles_to_interpolate, interval_indexes):
        """Interpolate the required angle components on the given time points

        Parameters
        ----------
        time_points : np.ndarray
            1D numpy array of length N of absolute time points
        angles_to_interpolate : List[int]
            C indexes of the angle components to interpolate
        interval_indexes : np.ndarray, optional
            intervals of the time axis where the given time points are expected

        Returns
        -------
        np.ndarray
            (C, N) numpy array of interpolated angle values
        """
        return self.interpolator.eval(
            time_points, interval_indexes, angles_to_interpolate
        )

    def get_yaw(self, time_points, interval_indexes=None) -> np.ndarray:
        """Return the yaw angles at the specified time points

        Parameters
        ----------
        time_points : np.ndarray
            1D numpy array of length N of absolute time points
        interval_indexes : np.ndarray, optional
            intervals of the time axis where the given time points are expected, by default None

        Returns
        -------
        np.ndarray
            1D numpy array of N yaw angles
        """
        return self._interpolate_angles(
            time_points, [self._ANGLE_INDEX["Y"]], interval_indexes
        )

    def get_pitch(self, time_points, interval_indexes=None) -> np.ndarray:
        """Return the pitch angles at the specified time points

        Parameters
        ----------
        time_points : np.ndarray
            1D numpy array of length N of absolute time points
        interval_indexes : np.ndarray, optional
            intervals of the time axis where the given time points are expected, by default None

        Returns
        -------
        np.ndarray
            1D numpy array of N yaw angles
        """
        return self._interpolate_angles(
            time_points, [self._ANGLE_INDEX["P"]], interval_indexes
        )

    def get_roll(self, time_points, interval_indexes=None) -> np.ndarray:
        """Return the roll angles at the specified time points

        Parameters
        ----------
        time_points : np.ndarray
            1D numpy array of length N of absolute time points
        interval_indexes : np.ndarray, optional
            intervals of the time axis where the given time points are expected, by default None

        Returns
        -------
        np.ndarray
            1D numpy array of N yaw angles
        """
        return self._interpolate_angles(
            time_points, [self._ANGLE_INDEX["R"]], interval_indexes
        )

    def get_arf(self, time_points: Union[PreciseDateTime, npt.ArrayLike]) -> np.ndarray:
        """Return the antenna reference frame matrix at the given time

        Parameters
        ----------
        time_points : npt.ArrayLike
            absolute time points as PreciseDateTime or an array like (N,) of PreciseDateTime

        Returns
        -------
        np.ndarray
            antenna reference frame as a (3, 3) or (N, 3, 3) numpy array
        """
        return compute_antenna_reference_frame(self._orbit, self, time_points)

    def sat2earthLOS(
        self,
        time_points: Union[PreciseDateTime, npt.ArrayLike],
        azimuth_angles: npt.ArrayLike,
        elevation_angles: npt.ArrayLike,
        *,
        altitude_over_wgs84: float = 0,
    ) -> np.ndarray:
        """Compute ground points illuminated with the given antenna patterns angles

        Parameters
        ----------
        time_points : npt.ArrayLike
            PreciseDateTime or (N,) array like of absolute times
        azimuth_angles, elevation_angles : npt.ArrayLike
            scalar or (N,) array like, in radians
        altitude_over_wgs84 : float, optional
            altitude over wgs84 ellipsoid, by default 0

        Returns
        -------
        np.ndarray
            ground points (3,) or (N, 3) numpy array
        """
        return direct_geocoding_with_pointing(
            self._orbit,
            self,
            time_points,
            azimuth_angles,
            elevation_angles,
            altitude_over_wgs84=altitude_over_wgs84,
        )

    def __repr__(self):
        axis_str = str(self._time_axis)
        ypr_str = str(self._ypr_angles)
        gso_str = str(self._orbit)

        axis_portion = (
            "Attitude defined on azimuth axis: " + os.linesep + axis_str + os.linesep
        )
        state_vectors_portion = (
            "Yaw Pitch Roll matrix: " + os.linesep + ypr_str + os.linesep
        )
        rotation_order = (
            f"Rotation order: {self.rotation_order.name.upper()}" + os.linesep
        )
        reference_frame = (
            f"Reference frame: {self.reference_frame.name.upper()}" + os.linesep
        )
        gso_portion = "Attitude info base on orbit:" + os.linesep + gso_str + os.linesep
        return (
            axis_portion
            + state_vectors_portion
            + rotation_order
            + reference_frame
            + gso_portion
        )


def compute_antenna_reference_frame(
    orbit: GeneralSarOrbit,
    attitude: GeneralSarAttitude,
    time_points: Union[PreciseDateTime, npt.ArrayLike],
) -> np.ndarray:
    """Return the antenna reference frame matrix at the given time

    Parameters
    ----------
    orbit : GeneralSarOrbit
        sensor orbit
    attitude : GeneralSarAttitude
        sensor attitude
    time_points : npt.ArrayLike
        one or more time points as a scalar or a (N,) array of PreciseDateTime

    Returns
    -------
    np.ndarray
        antenna reference frame as a (3,3) or (N, 3, 3) numpy array
    """
    time_points = np.asarray(time_points)

    # Workaround to bypass get_position, get_velocity, get_yaw, get_pitch, get_roll inconsistent interfaces
    time_points_as_1d_array = time_points.reshape((-1,))
    shape_1d = time_points.shape
    shape_2d = time_points.shape + (3,)

    sensor_position = orbit.get_position(time_points_as_1d_array).T.reshape(shape_2d)
    sensor_velocity = orbit.get_velocity(time_points_as_1d_array).T.reshape(shape_2d)

    initial_frame = compute_sensor_local_axis(
        sensor_position, sensor_velocity, attitude.reference_frame
    )

    return compute_antenna_reference_frame_from_euler_angles(
        order=attitude.rotation_order,
        initial_reference_frame_axis=initial_frame,
        yaw=np.deg2rad(attitude.get_yaw(time_points_as_1d_array).reshape(shape_1d)),
        pitch=np.deg2rad(attitude.get_pitch(time_points_as_1d_array).reshape(shape_1d)),
        roll=np.deg2rad(attitude.get_roll(time_points_as_1d_array).reshape(shape_1d)),
    )


def compute_pointing_directions(
    antenna_reference_frames: npt.ArrayLike,
    azimuth_angles: npt.ArrayLike,
    elevation_angles: npt.ArrayLike,
) -> np.ndarray:
    """Compute the pointing directions corresponding to the given angles

    Parameters
    ----------
    antenna_reference_frames : npt.ArrayLike
        (3, 3) or (N, 3, 3) array like
    azimuth_angles, elevation_angles : npt.ArrayLike
        scalar or (N,) array like

    Returns
    -------
    np.ndarray
        (3,) or (N, 3) numpy array with pointing directions

    Raises
    ------
    ValueError
        in case of invalid input
    """
    antenna_reference_frames = np.asarray(antenna_reference_frames)
    azimuth_angles = np.asarray(azimuth_angles)
    elevation_angles = np.asarray(elevation_angles)

    if antenna_reference_frames.shape[-2:] != (3, 3):
        raise ValueError(
            f"Invalid antenna_reference_frames shape: {antenna_reference_frames.shape} "
            + "should be either (3,3) or (N,3,3)"
        )

    if (
        azimuth_angles.size > 1
        and antenna_reference_frames.size > 9
        and azimuth_angles.size * 9 != antenna_reference_frames.size
    ):
        raise ValueError(
            "Incompatible azimuth_angles and antenna_reference_frames shapes: "
            + f"{azimuth_angles.shape}, {antenna_reference_frames.shape}"
        )

    if (
        elevation_angles.size > 1
        and antenna_reference_frames.size > 9
        and elevation_angles.size * 9 != antenna_reference_frames.size
    ):
        raise ValueError(
            "Incompatible elevation_angles and antenna_reference_frames shapes: "
            + f"{elevation_angles.shape}, {antenna_reference_frames.shape}"
        )

    if (
        azimuth_angles.size > 1
        and elevation_angles.size > 1
        and azimuth_angles.size != elevation_angles.size
    ):
        raise ValueError(
            f"Incompatible azimuth_angles and elevation_angles shapes: {azimuth_angles.shape}, {elevation_angles.shape}"
        )

    if azimuth_angles.shape != elevation_angles.shape:
        broadcast_shape = np.broadcast_shapes(
            azimuth_angles.shape, elevation_angles.shape
        )
        azimuth_angles = np.broadcast_to(azimuth_angles, broadcast_shape)
        elevation_angles = np.broadcast_to(elevation_angles, broadcast_shape)

    ux = np.tan(azimuth_angles)
    uy = np.tan(elevation_angles)
    uz = np.ones_like(ux)
    local_directions = np.stack([ux, uy, uz], axis=-1)
    local_directions = local_directions / np.linalg.norm(
        local_directions, axis=-1, keepdims=True
    )

    return np.einsum("...jk,...k->...j", antenna_reference_frames, local_directions)


def direct_geocoding_with_pointing(
    orbit: GeneralSarOrbit,
    attitude: GeneralSarAttitude,
    time_points: Union[PreciseDateTime, npt.ArrayLike],
    azimuth_angles: npt.ArrayLike,
    elevation_angles: npt.ArrayLike,
    *,
    altitude_over_wgs84: float = 0.0,
) -> np.ndarray:
    """Compute ground points illuminated with the given antenna patterns angles

    Parameters
    ----------
    orbit : GeneralSarOrbit
        sensor orbit
    attitude : GeneralSarAttitude
        sensor attitude
    time_points : npt.ArrayLike
        PreciseDateTime or (N,) array like of absolute times
    azimuth_angles, elevation_angles : npt.ArrayLike
        scalar or (N,) array like, in radians
    altitude_over_wgs84 : float, optional
        altitude over wgs84 ellipsoi, by default 0.0

    Returns
    -------
    np.ndarray
        ground points (3,) or (N, 3) numpy array

    Raises
    ------
    ValueError
        in case of invalid input
    """
    time_points = np.asarray(time_points)
    azimuth_angles = np.asarray(azimuth_angles)
    elevation_angles = np.asarray(elevation_angles)

    num_elements = max(time_points.size, azimuth_angles.size, elevation_angles.size)
    if (
        time_points.size not in (1, num_elements)
        or azimuth_angles.size not in (1, num_elements)
        or elevation_angles.size not in (1, num_elements)
    ):
        raise ValueError(
            "Incompatible time_points, azimuth_angles and elevation_angles shapes: "
            + f"{time_points.shape}, {azimuth_angles.shape} and {elevation_angles.shape}"
        )

    sensor_positions = orbit.get_position(time_points.reshape((-1,))).T.reshape(
        time_points.shape + (3,)
    )
    antenna_reference_frames = attitude.get_arf(time_points)
    pointing_directions = compute_pointing_directions(
        antenna_reference_frames, azimuth_angles, elevation_angles
    )
    return direct_geocoding_with_looking_direction(
        sensor_positions, pointing_directions, altitude_over_wgs84=altitude_over_wgs84
    )


def create_general_sar_attitude(
    state_vectors: StateVectors,
    attitude_info: AttitudeInfo,
    ignore_anx_after_orbit_start=False,
) -> GeneralSarAttitude:
    """Create general sar attitude object from state vectors and attitude info metadata

    Parameters
    ----------
    state_vectors : StateVectors
        state vectors as a StateVectors metadata object
    attitude_info : AttitudeInfo
        attitude data as a AttitudeInfo metadata object
    ignore_anx_after_orbit_start : bool, default False
        if true, the ANX time in state_vectors is ignored in case it is not immediately
        antecedent to the orbit start

    Returns
    -------
    GeneralSarAttitude
        the new GeneralSarAttitude object
    """
    gso = create_general_sar_orbit(
        state_vectors, ignore_anx_after_orbit_start=ignore_anx_after_orbit_start
    )

    if (
        attitude_info.time_step is None
        or attitude_info.attitude_records_number is None
        or attitude_info.reference_time is None
    ):
        raise ValueError(
            "Cannot create general sar attitude: incomplete attitude information"
        )

    time_axis = are_ax.RegularAxis(
        (0, attitude_info.time_step, attitude_info.attitude_records_number),
        attitude_info.reference_time,
    )
    ypr_matrix = np.vstack(
        (
            attitude_info.yaw_vector,
            attitude_info.pitch_vector,
            attitude_info.roll_vector,
        )
    )
    gsa = GeneralSarAttitude(
        gso,
        time_axis,
        ypr_matrix,
        attitude_info.rotation_order.value,
        attitude_info.reference_frame.value,
    )
    return gsa
