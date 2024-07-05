# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Attitude-related utilities
--------------------------
"""
import numpy as np
import numpy.typing as npt
from scipy.spatial import transform

from arepytools.geometry.reference_frames import ReferenceFrameLike
from arepytools.geometry.rotation import (
    RotationOrderLike,
    compute_euler_angles_from_rotation,
    compute_rotation,
)


def compute_antenna_reference_frame_from_euler_angles(
    order: RotationOrderLike,
    initial_reference_frame_axis: np.ndarray,
    yaw: npt.ArrayLike,
    pitch: npt.ArrayLike,
    roll: npt.ArrayLike,
) -> np.ndarray:
    """Computing the Antenna Reference Frame (ARF) from euler angles (YAW, PITCH and ROLL) giving a rotation order and
    a reference frame.

    Parameters
    ----------
    order : RotationOrderLike
        rotation order for the euler angles
    initial_reference_frame_axis : np.ndarray
        reference frame axis of the sensor
    yaw : npt.ArrayLike
        sensor's yaw
    pitch : npt.ArrayLike
        sensor's pitch
    roll : npt.ArrayLike
        sensor's roll

    Returns
    -------
    np.ndarray
        antenna reference frame for the sensor
    """

    rotation = compute_rotation(order=order, yaw=yaw, pitch=pitch, roll=roll)

    return np.matmul(initial_reference_frame_axis, rotation.as_matrix())


def compute_euler_angles_from_antenna_reference_frame(
    initial_reference_frame_axis: np.ndarray,
    antenna_reference_frame: np.ndarray,
    order: RotationOrderLike,
) -> np.ndarray:
    """Compute euler angles (YAW, PITCH and ROLL) from Antenna Reference Frame (ARF), the initial reference frame and
    rotation order.

    Parameters
    ----------
    initial_reference_frame_axis : np.ndarray
        initial reference frame axis of the sensor, (3, 3) or (N, 3, 3)
    antenna_reference_frame : np.ndarray
        antenna reference frame of the sensor, (3, 3) or (N, 3, 3)
    order : RotationOrderLike
        rotation order

    Returns
    -------
    np.ndarray
        euler angles array, (N, 3), columns being in the same rotation order provided as input
    """
    if initial_reference_frame_axis.shape != antenna_reference_frame.shape:
        raise RuntimeError(
            f"input shape mismatch: init ref frame {initial_reference_frame_axis.shape} != arf {antenna_reference_frame.shape}"
        )
    init_ref_frame = (
        np.transpose(initial_reference_frame_axis, (0, 2, 1))
        if initial_reference_frame_axis.ndim == 3
        else initial_reference_frame_axis.T
    )
    rotation = transform.Rotation.from_matrix(
        np.matmul(init_ref_frame, antenna_reference_frame)
    )

    return compute_euler_angles_from_rotation(order=order, rotation=rotation)
