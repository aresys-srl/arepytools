# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Create Orbit object from State Vectors
--------------------------------------
"""
import numpy as np

from arepytools.geometry.orbit import Orbit
from arepytools.io.metadata import StateVectors


def create_orbit(state_vectors: StateVectors) -> Orbit:
    """Creating a Orbit object from metadata StateVectors.

    Parameters
    ----------
    state_vectors : StateVectors
        product metadata StateVectors

    Returns
    -------
    Orbit
        interpolated Orbit object from given StateVectors
    """
    time_axis = (
        np.arange(
            0,
            state_vectors.number_of_state_vectors * state_vectors.time_step,
            state_vectors.time_step,
        )
        + state_vectors.reference_time
    )
    return Orbit(
        times=time_axis,
        positions=state_vectors.position_vector.reshape(-1, 3),
        velocities=state_vectors.velocity_vector.reshape(-1, 3),
    )
