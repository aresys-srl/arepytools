# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Rotation module
---------------
"""

from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.spatial import transform


class RotationOrder(Enum):
    """Yaw / Pitch / Roll rotation orders"""

    ypr = "YPR"
    yrp = "YRP"
    pry = "PRY"
    pyr = "PYR"
    ryp = "RYP"
    rpy = "RPY"


RotationOrderLike = Union[str, RotationOrder]
""":class:`RotationOrder` like type hint"""


def compute_rotation(
    order: RotationOrderLike,
    *,
    yaw: npt.ArrayLike,
    pitch: npt.ArrayLike,
    roll: npt.ArrayLike,
) -> transform.Rotation:
    """Compute the rotation defined by a rotation order and the yaw pitch roll angles

    Parameters
    ----------
    order : RotationOrderLike
        rotation order
    yaw, pitch, roll: npt.ArrayLike
        scalar or (N,) angles in radians

    Returns
    -------
    transform.Rotation
        a (stack of) rotation objects equivalent to a (3, 3) or (N, 3, 3) np.ndarray

    Examples
    --------

    single rotation

    >>> rotation = compute_rotation(RotationOrder.ypr, yaw=0.0, pitch=0.0, roll=np.deg2rad(30.0))
    >>> print(rotation.as_matrix())
    [[ 1.         0.         0.       ]
     [ 0.         0.8660254 -0.5      ]
     [ 0.         0.5        0.8660254]]

    multiple rotation

    >>> roll = np.deg2rad(np.arange(10,26,5, dtype=float))
    >>> rotation = compute_rotation(RotationOrder.ypr, yaw=np.zeros_like(roll), pitch=np.zeros_like(roll), roll=roll)
    >>> print(rotation.as_matrix().shape)
     (4, 3, 3)

    rotation order as string

    >>> compute_rotation("YPR", yaw=0.0, pitch=0.0, roll=np.deg2rad(30.0))
    """
    order = RotationOrder(order)

    # upper case / lower case axis character matters.
    translation_table = str.maketrans({"Y": "Z", "P": "Y", "R": "X"})
    euler_sequence = order.value.translate(translation_table)

    angles = {"Y": yaw, "P": pitch, "R": roll}
    euler_angles = np.stack([angles[axis] for axis in order.value], axis=-1)

    return transform.Rotation.from_euler(euler_sequence, euler_angles)


__all__ = ["RotationOrder", "compute_rotation"]
