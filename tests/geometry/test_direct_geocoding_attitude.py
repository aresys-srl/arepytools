# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/direct_geocoding.py direct_geocoding_attitude functionality"""

import unittest

import numpy as np
from scipy.constants import speed_of_light

from arepytools.geometry.direct_geocoding import (
    GeocodingSide,
    _doppler_equation,
    _ellipse_equation,
    direct_geocoding_attitude,
    direct_geocoding_monostatic_attitude_init,
)
from arepytools.geometry.ellipsoid import WGS84
from arepytools.geometry.generalsarattitude import GeneralSarAttitude
from arepytools.geometry.generalsarorbit import GeneralSarOrbit
from arepytools.io.metadata import PreciseDateTime
from arepytools.math.axis import RegularAxis

state_vectors = np.array(
    [
        4401464.72398884,
        764575.42517419,
        4539105.27862944,
        4400759.95127639,
        764452.999588603,
        4539804.48018504,
        4400055.07025084,
        764330.555188024,
        4540503.56953453,
        4399350.08096448,
        764208.091981531,
        4541202.54662631,
        4398644.98346958,
        764085.609978207,
        4541901.4114088,
        4397939.77781846,
        763963.10918714,
        4542600.16383045,
        4397234.46406345,
        763840.589617422,
        4543298.80383973,
        4396529.04225692,
        763718.051278147,
        4543997.33138513,
        4395823.51245126,
        763595.494178416,
        4544695.74641519,
    ]
)

time_axis = RegularAxis(
    (0, 1, state_vectors.size // 3),
    PreciseDateTime.from_utc_string("17-FEB-2020 16:06:50.906728901874"),
)


_yaw = np.array(
    [
        -10.9353126825047,
        -10.9353126825047,
        -10.9353126825047,
        -10.9353126825047,
        -10.9353126825047,
        -10.9353126825047,
        -10.9353126825047,
        -10.9353126825047,
        -10.9353126825047,
    ]
)
_pitch = np.array(
    [
        -19.9452555208155,
        -19.9452555208155,
        -19.9452555208155,
        -19.9452555208155,
        -19.9452555208155,
        -19.9452555208155,
        -19.9452555208155,
        -19.9452555208155,
        -19.9452555208155,
    ]
)
_roll = np.array(
    [
        -57.1252699331354,
        -57.1252699331354,
        -57.1252699331354,
        -57.1252699331354,
        -57.1252699331354,
        -57.1252699331354,
        -57.1252699331354,
        -57.1252699331354,
        -57.1252699331354,
    ]
)

ypr_matrix = np.vstack((_yaw, _pitch, _roll))


def _doppler_equation_residual(
    sensor_pos: np.ndarray,
    sensor_vel: np.ndarray,
    ground_points: np.ndarray,
    wavelength: float,
    doppler_freq: float,
) -> np.ndarray:
    """Evaluating doppler equation residual.

    Parameters
    ----------
    sensor_pos : np.ndarray
        sensor position, (3,) or (N, 3)
    sensor_vel : np.ndarray
        sensor velocity, (3,) or (N, 3)
    ground_points : np.ndarray
        ground points from direct geocoding solution, (3,) or (N, 3)
    wavelength : float
        carrier signal wavelength
    doppler_freq : float
        doppler frequency

    Returns
    -------
    np.ndarray
        doppler equation residual
    """

    los = sensor_pos - ground_points
    los_vel_prod = np.sum(sensor_vel * los, axis=-1)
    distance = np.sqrt(np.sum(los * los, axis=-1))

    doppler_residual, _ = _doppler_equation(
        pv_scalar=los_vel_prod,
        sat2point=los,
        sat_velocity=sensor_vel,
        distance=distance,
        wavelength=wavelength,
        frequency_doppler_centroid=doppler_freq,
    )
    return np.array(doppler_residual)


def _range_equation_residual(
    sensor_pos: np.ndarray, ground_points: np.ndarray, range_time: float
) -> np.ndarray:
    """Evaluating range equation residual.

    Parameters
    ----------
    sensor_pos : np.ndarray
        sensor position, (3,) or (N, 3)
    ground_points : np.ndarray
        ground points from direct geocoding solution, (3,) or (N, 3)
    range_time : float
        range time

    Returns
    -------
    np.ndarray
        range equation residual
    """

    rng_dst = speed_of_light * range_time / 2.0
    los = sensor_pos - ground_points
    distance = np.sqrt(np.sum(los * los, axis=-1))

    rng_residual = (distance - rng_dst) / speed_of_light

    return np.array(rng_residual)


def _ellipse_equation_residual(ground_points: np.ndarray) -> np.ndarray:
    """Ellipse equation residual.

    Parameters
    ----------
    ground_points : np.ndarray
        ground points from direct geocoding solution, (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        ellipse equation residual
    """
    r_ep2 = WGS84.semi_minor_axis**2
    r_ee2 = WGS84.semi_major_axis**2

    ellipse_residual = _ellipse_equation(ground_points, r_ee2, r_ep2)

    return ellipse_residual


def _get_arf_velocity(sensor_vel: np.ndarray, arf: np.ndarray) -> np.ndarray:
    """Get scaled_arf_velocities from inputs.

    Parameters
    ----------
    sensor_vel : np.ndarray
        sensor velocity, (3,) or (N, 3)
    arf : np.ndarray
        antenna reference frame full matrix, in the form (3, 3) or (N, 3, 3)

    Returns
    -------
    np.ndarray
        scaled arf velocities
    """

    antenna_reference_frames = np.asarray(arf)
    sensor_vel = np.asarray(sensor_vel)
    normalized_velocity = np.linalg.norm(sensor_vel, axis=-1, keepdims=True)
    scaled_arf_velocities = normalized_velocity * antenna_reference_frames[..., :, 0]

    return scaled_arf_velocities


class DirectGeocodingAttitudeTest(unittest.TestCase):
    """Testing direct geocoding with attitude"""

    def setUp(self):
        """Setting up variables for testing"""

        self.orbit = GeneralSarOrbit(time_axis, state_vectors)
        self.attitude = GeneralSarAttitude(
            self.orbit, time_axis, ypr_matrix, "YPR", "ZERODOPPLER"
        )

        self.rng_times = 2.05624579e-05
        self.az_times = PreciseDateTime.from_utc_string(
            "17-FEB-2020 16:07:10.906728901874"
        )
        self.init_guess = np.array(
            [4385932.628762595, 764443.4718341012, 4551945.624046889]
        )

        self.sat_pos = self.orbit.get_position(self.az_times).T.squeeze()
        self.sat_vel = self.orbit.get_velocity(self.az_times).T.squeeze()
        self.arf = self.attitude.get_arf(self.az_times)
        self.M = 4
        self.N = 5
        self.side_looking = "RIGHT"

        self.tolerance = 1e-5
        self.residual_tolerance = 1e-8
        self.results_ref = np.array(
            [4385882.195057692, 764600.9869913795, 4551967.6143934],
        )

    def test_direct_geocoding_attitude_case0a(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 0a"""

        # case 0a: 1 sensor_pos/vel, 1 arf, 1 range time
        positions = direct_geocoding_attitude(
            sensor_positions=self.sat_pos,
            sensor_velocities=self.sat_vel,
            antenna_reference_frames=self.arf,
            range_times=self.rng_times,
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.sat_pos,
            sensor_vel=_get_arf_velocity(self.sat_vel, self.arf),
            ground_points=positions,
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.sat_pos, ground_points=positions, range_time=self.rng_times
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions)

        self.assertEqual(positions.ndim, 1)
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions, self.results_ref, atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_attitude_case0b(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 0b"""

        # case 0b: [1] sensor_pos/vel, 1 arf, [1] range time
        positions = direct_geocoding_attitude(
            sensor_positions=self.sat_pos.reshape(1, 3),
            sensor_velocities=self.sat_vel.reshape(1, 3),
            antenna_reference_frames=self.arf,
            range_times=[self.rng_times],
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.sat_pos,
            sensor_vel=_get_arf_velocity(self.sat_vel, self.arf),
            ground_points=positions,
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.sat_pos, ground_points=positions, range_time=self.rng_times
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions)

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions, self.results_ref.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_attitude_case0c(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 0c"""

        # case 0c: [1] sensor_pos/vel, 1 arf, 1 range time
        positions = direct_geocoding_attitude(
            sensor_positions=self.sat_pos.reshape(1, 3),
            sensor_velocities=self.sat_vel.reshape(1, 3),
            antenna_reference_frames=self.arf,
            range_times=self.rng_times,
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.sat_pos,
            sensor_vel=_get_arf_velocity(self.sat_vel, self.arf),
            ground_points=positions,
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.sat_pos, ground_points=positions, range_time=self.rng_times
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions)

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions, self.results_ref.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_attitude_case1(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 1"""

        # case 1: 1 sensor_pos/vel, 1 arf, M range times
        positions = direct_geocoding_attitude(
            sensor_positions=self.sat_pos,
            sensor_velocities=self.sat_vel,
            antenna_reference_frames=self.arf,
            range_times=np.repeat(self.rng_times, self.M),
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.sat_pos,
            sensor_vel=_get_arf_velocity(self.sat_vel, self.arf),
            ground_points=positions,
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.sat_pos, ground_points=positions, range_time=self.rng_times
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions)

        self.assertEqual(positions.ndim, 2)
        self.assertEqual(positions.shape, (self.M, 3))
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions,
            np.full((self.M, 3), self.results_ref),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_case2(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 2"""

        # case 2: N sensor_pos/vel, 1 arf, 1 range time
        positions = direct_geocoding_attitude(
            sensor_positions=np.full((self.N, 3), self.sat_pos),
            sensor_velocities=np.full((self.N, 3), self.sat_vel),
            antenna_reference_frames=self.arf,
            range_times=self.rng_times,
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            sensor_vel=_get_arf_velocity(np.full((self.N, 3), self.sat_vel), self.arf),
            ground_points=positions,
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            ground_points=positions,
            range_time=self.rng_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions)

        self.assertEqual(positions.ndim, 2)
        self.assertEqual(positions.shape, (self.N, 3))
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions,
            np.full((self.N, 3), self.results_ref),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_case3(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 3"""

        # case 3: N sensor_pos/vel, N arf, 1 range time
        positions = direct_geocoding_attitude(
            sensor_positions=np.full((self.N, 3), self.sat_pos),
            sensor_velocities=np.full((self.N, 3), self.sat_vel),
            antenna_reference_frames=np.full((self.N, 3, 3), self.arf),
            range_times=self.rng_times,
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            sensor_vel=_get_arf_velocity(
                np.full((self.N, 3), self.sat_vel), np.full((self.N, 3, 3), self.arf)
            ),
            ground_points=positions,
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            ground_points=positions,
            range_time=self.rng_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions)

        self.assertEqual(positions.ndim, 2)
        self.assertEqual(positions.shape, (self.N, 3))
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions,
            np.full((self.N, 3), self.results_ref),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_case4(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 4"""

        # case 4: N sensor_pos/vel, 1 arf, M range times
        positions = direct_geocoding_attitude(
            sensor_positions=np.full((self.N, 3), self.sat_pos),
            sensor_velocities=np.full((self.N, 3), self.sat_vel),
            antenna_reference_frames=self.arf,
            range_times=np.repeat(self.rng_times, self.M),
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            sensor_vel=_get_arf_velocity(np.full((self.N, 3), self.sat_vel), self.arf),
            ground_points=positions[:, 0, :],
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            ground_points=positions[:, 0, :],
            range_time=self.rng_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions[:, 0, :])

        self.assertEqual(positions.ndim, 3)
        self.assertEqual(positions.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions,
            np.full((self.N, self.M, 3), self.results_ref),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_case5(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 5"""

        # case 5: N sensor_pos/vel, N arf, M range times
        positions = direct_geocoding_attitude(
            sensor_positions=np.full((self.N, 3), self.sat_pos),
            sensor_velocities=np.full((self.N, 3), self.sat_vel),
            antenna_reference_frames=np.full((self.N, 3, 3), self.arf),
            range_times=np.repeat(self.rng_times, self.M),
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            sensor_vel=_get_arf_velocity(
                np.full((self.N, 3), self.sat_vel), np.full((self.N, 3, 3), self.arf)
            ),
            ground_points=positions[:, 0, :],
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            ground_points=positions[:, 0, :],
            range_time=self.rng_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions[:, 0, :])

        self.assertEqual(positions.ndim, 3)
        self.assertEqual(positions.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions,
            np.full((self.N, self.M, 3), self.results_ref),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_case6(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 6"""

        # case 6: 1 sensor_pos/vel, N arf, 1 range time
        positions = direct_geocoding_attitude(
            sensor_positions=self.sat_pos,
            sensor_velocities=self.sat_vel,
            antenna_reference_frames=np.full((self.N, 3, 3), self.arf),
            range_times=self.rng_times,
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            sensor_vel=_get_arf_velocity(
                np.full((self.N, 3), self.sat_vel), np.full((self.N, 3, 3), self.arf)
            ),
            ground_points=positions,
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            ground_points=positions,
            range_time=self.rng_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions)

        self.assertEqual(positions.ndim, 2)
        self.assertEqual(positions.shape, (self.N, 3))
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions,
            np.full((self.N, 3), self.results_ref),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_case7(self) -> None:
        """Testing direct_geocoding_attitude from geometry/direct_geocoding.py submodule, case 7"""

        # case 7: 1 sensor_pos/vel, N arf, M range
        positions = direct_geocoding_attitude(
            sensor_positions=self.sat_pos,
            sensor_velocities=self.sat_vel,
            antenna_reference_frames=np.full((self.N, 3, 3), self.arf),
            range_times=np.repeat(self.rng_times, self.M),
            geocoding_side=self.side_looking,
            initial_guesses=self.init_guess,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            sensor_vel=_get_arf_velocity(
                np.full((self.N, 3), self.sat_vel), np.full((self.N, 3, 3), self.arf)
            ),
            ground_points=positions[:, 0, :],
            doppler_freq=0,
            wavelength=1,
        )
        range_residual = _range_equation_residual(
            sensor_pos=np.full((self.N, 3), self.sat_pos),
            ground_points=positions[:, 0, :],
            range_time=self.rng_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=positions[:, 0, :])

        self.assertEqual(positions.ndim, 3)
        self.assertEqual(positions.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            range_residual,
            np.zeros_like(range_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            ellipse_residual,
            np.zeros_like(ellipse_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            positions,
            np.full((self.N, self.M, 3), self.results_ref),
            atol=self.tolerance,
            rtol=0,
        )


class DirectGeocodingAttitudeInitTest(unittest.TestCase):
    """Testing direct geocoding with attitude init"""

    def setUp(self):
        """Setting up variables for testing"""

        self.orbit = GeneralSarOrbit(time_axis, state_vectors)
        self.attitude = GeneralSarAttitude(
            self.orbit, time_axis, ypr_matrix, "YPR", "ZERODOPPLER"
        )

        self.rng_times = [2.05624579e-05, 2.05634579e-05, 2.05644579e-05]
        self.az_times = [
            PreciseDateTime.from_utc_string("17-FEB-2020 16:07:10.906728901874")
        ] * 2

        self.sat_pos = self.orbit.get_position(self.az_times).T
        self.sat_vel = self.orbit.get_velocity(self.az_times).T
        self.arf = self.attitude.get_arf(self.az_times)

        self.side_looking = GeocodingSide.RIGHT_LOOKING

        self.tolerance = 1e-5
        self.results_ref_no_pert = np.array(
            [4385932.628762595, 764443.4718341012, 4551945.624046889]
        )
        self.results_ref_pert = np.array(
            [4385915.514128459, 764541.9963960052, 4551945.624046803]
        )

    def test_direct_geocoding_attitude_init_case0a(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 0a"""

        # case 0a: 1 sensor_pos (3,), 1 sensor vel (3,), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0],
            sensor_velocities=self.sat_vel[0],
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
        )

        self.assertEqual(positions.ndim, 1)
        np.testing.assert_allclose(
            positions, self.results_ref_no_pert, atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_attitude_init_case0a_pert(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 0a"""

        # case 0a: 1 sensor_pos (3,), 1 sensor vel (3,), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0],
            sensor_velocities=self.sat_vel[0],
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
            perturbation_for_nadir_geom=True,
        )

        self.assertEqual(positions.ndim, 1)
        np.testing.assert_allclose(
            positions, self.results_ref_pert, atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_attitude_init_case0b(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 0b"""

        # case 0b: 1 sensor_pos (1, 3), 1 sensor vel (1, 3), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0].reshape(1, 3),
            sensor_velocities=self.sat_vel[0].reshape(1, 3),
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            self.results_ref_no_pert.reshape(1, 3),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case0b_pert(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 0b"""

        # case 0b: 1 sensor_pos (1, 3), 1 sensor vel (1, 3), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0].reshape(1, 3),
            sensor_velocities=self.sat_vel[0].reshape(1, 3),
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
            perturbation_for_nadir_geom=True,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            self.results_ref_pert.reshape(1, 3),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case0c(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 0c"""

        # case 0c: 1 sensor_pos (1, 3), 1 sensor vel (3,), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0].reshape(1, 3),
            sensor_velocities=self.sat_vel[0],
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            self.results_ref_no_pert.reshape(1, 3),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case0c_pert(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 0c"""

        # case 0c: 1 sensor_pos (1, 3), 1 sensor vel (3,), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0].reshape(1, 3),
            sensor_velocities=self.sat_vel[0],
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
            perturbation_for_nadir_geom=True,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            self.results_ref_pert.reshape(1, 3),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case0d(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 0d"""

        # case 0d: 1 sensor_pos (3,), 1 sensor vel (1, 3), 1 arf

        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0],
            sensor_velocities=self.sat_vel[0].reshape(1, 3),
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            self.results_ref_no_pert.reshape(1, 3),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case0d_pert(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 0d"""

        # case 0d: 1 sensor_pos (3,), 1 sensor vel (1, 3), 1 arf

        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0],
            sensor_velocities=self.sat_vel[0].reshape(1, 3),
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
            perturbation_for_nadir_geom=True,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            self.results_ref_pert.reshape(1, 3),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case1a(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 1a"""

        # case 1a: N sensor_pos (N, 3), N sensor vel (N, 3), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos,
            sensor_velocities=self.sat_vel,
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            np.full(self.sat_pos.shape, self.results_ref_no_pert),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case1a_pert(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 1a"""

        # case 1a: N sensor_pos (N, 3), N sensor vel (N, 3), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos,
            sensor_velocities=self.sat_vel,
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
            perturbation_for_nadir_geom=True,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            np.full(self.sat_pos.shape, self.results_ref_pert),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case1b(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 1b"""

        # case 1b: N sensor_pos (N, 3), 1 sensor vel (3,), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos,
            sensor_velocities=self.sat_vel[0],
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            np.full(self.sat_pos.shape, self.results_ref_no_pert),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case1b_pert(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 1b"""

        # case 1b: N sensor_pos (N, 3), 1 sensor vel (3,), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos,
            sensor_velocities=self.sat_vel[0],
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
            perturbation_for_nadir_geom=True,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            np.full(self.sat_pos.shape, self.results_ref_pert),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case1c(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 1c"""

        # case 1c: 1 sensor_pos (3,), N sensor vel (N, 3), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0],
            sensor_velocities=self.sat_vel,
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            np.full(self.sat_pos.shape, self.results_ref_no_pert),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case1c_pert(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 1c"""

        # case 1c: 1 sensor_pos (3,), N sensor vel (N, 3), 1 arf
        positions = direct_geocoding_monostatic_attitude_init(
            sensor_positions=self.sat_pos[0],
            sensor_velocities=self.sat_vel,
            antenna_reference_frames=self.arf[0],
            geocoding_side=self.side_looking,
            perturbation_for_nadir_geom=True,
        )

        self.assertEqual(positions.ndim, 2)
        np.testing.assert_allclose(
            positions,
            np.full(self.sat_pos.shape, self.results_ref_pert),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_attitude_init_case2a(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 2a"""

        # case 2a: N sensor_pos (N, 3), N sensor vel (N, 3), N arf
        for flag in True, False:
            # testing both condition with perturbation on and off
            positions = direct_geocoding_monostatic_attitude_init(
                sensor_positions=self.sat_pos,
                sensor_velocities=self.sat_vel,
                antenna_reference_frames=self.arf,
                geocoding_side=self.side_looking,
                perturbation_for_nadir_geom=flag,
            )

            self.assertEqual(positions.ndim, 2)
            if flag:
                np.testing.assert_allclose(
                    positions,
                    np.full(self.sat_pos.shape, self.results_ref_pert),
                    atol=self.tolerance,
                    rtol=0,
                )
            else:
                np.testing.assert_allclose(
                    positions,
                    np.full(self.sat_pos.shape, self.results_ref_no_pert),
                    atol=self.tolerance,
                    rtol=0,
                )

    def test_direct_geocoding_attitude_init_case3a(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 3a"""

        # case 3a: 1 sensor_pos (3,), 1 sensor vel (3,), N arf
        for flag in True, False:
            # testing both condition with perturbation on and off
            positions = direct_geocoding_monostatic_attitude_init(
                sensor_positions=self.sat_pos[0],
                sensor_velocities=self.sat_vel[0],
                antenna_reference_frames=self.arf,
                geocoding_side=self.side_looking,
                perturbation_for_nadir_geom=flag,
            )

            self.assertEqual(positions.ndim, 2)
            if flag:
                np.testing.assert_allclose(
                    positions,
                    np.full(self.sat_pos.shape, self.results_ref_pert),
                    atol=self.tolerance,
                    rtol=0,
                )
            else:
                np.testing.assert_allclose(
                    positions,
                    np.full(self.sat_pos.shape, self.results_ref_no_pert),
                    atol=self.tolerance,
                    rtol=0,
                )

    def test_direct_geocoding_attitude_init_case3b(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 3b"""

        # case 3b: N sensor_pos (N, 3), 1 sensor vel (3,), N arf
        for flag in True, False:
            # testing both condition with perturbation on and off
            positions = direct_geocoding_monostatic_attitude_init(
                sensor_positions=self.sat_pos,
                sensor_velocities=self.sat_vel[0],
                antenna_reference_frames=self.arf,
                geocoding_side=self.side_looking,
                perturbation_for_nadir_geom=flag,
            )

            self.assertEqual(positions.ndim, 2)
            if flag:
                np.testing.assert_allclose(
                    positions,
                    np.full(self.sat_pos.shape, self.results_ref_pert),
                    atol=self.tolerance,
                    rtol=0,
                )
            else:
                np.testing.assert_allclose(
                    positions,
                    np.full(self.sat_pos.shape, self.results_ref_no_pert),
                    atol=self.tolerance,
                    rtol=0,
                )

    def test_direct_geocoding_attitude_init_case3c(self) -> None:
        """Testing direct_geocoding_monostatic_attitude_init, case 3c"""

        # case 3c: 1 sensor_pos (3,), N sensor vel (N, 3), N arf
        for flag in True, False:
            # testing both condition with perturbation on and off
            positions = direct_geocoding_monostatic_attitude_init(
                sensor_positions=self.sat_pos[0],
                sensor_velocities=self.sat_vel,
                antenna_reference_frames=self.arf,
                geocoding_side=self.side_looking,
                perturbation_for_nadir_geom=flag,
            )

            self.assertEqual(positions.ndim, 2)
            if flag:
                np.testing.assert_allclose(
                    positions,
                    np.full(self.sat_pos.shape, self.results_ref_pert),
                    atol=self.tolerance,
                    rtol=0,
                )
            else:
                np.testing.assert_allclose(
                    positions,
                    np.full(self.sat_pos.shape, self.results_ref_no_pert),
                    atol=self.tolerance,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
