# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/direct_geocoding.py direct_geocoding_monostatic functionalities"""

import unittest

import numpy as np
from scipy.constants import speed_of_light

from arepytools.geometry.direct_geocoding import (
    AmbiguousInputCorrelation,
    GeocodingSide,
    _direct_geocoding_monostatic_core,
    _doppler_equation,
    _ellipse_equation,
    _newton_for_direct_geocoding_monostatic,
    direct_geocoding_monostatic,
    direct_geocoding_monostatic_init,
)
from arepytools.geometry.ellipsoid import WGS84


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


class DirectGeocodingMonostaticTest(unittest.TestCase):
    """Testing direct geocoding monostatic"""

    def setUp(self):
        self.positions = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004],
        )
        self.scaled_arf_velocities = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346],
        )
        self.initial_guesses = np.array(
            [4385932.628762595, 764443.4718341012, 4551945.624046889]
        )
        self.range_times = np.array([2.05624579e-05])
        self.doppler_freqs = 0
        self.geodetic_altitude = 0
        self.look_direction = GeocodingSide.RIGHT_LOOKING
        self.wavelength = 1
        self.N = 4
        self.M = 5

        self.tolerance = 1e-5
        self.residual_tolerance = 1e-8

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def test_direct_geocoding_monostatic_case0a(self) -> None:
        """Testing direct geocoding monostatic function, case 0a"""

        # case 0a: 1 pos (3,), 1 vel (3,), 1 rng time (float), 1 initial guess (3,)
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions,
            sensor_velocities=self.scaled_arf_velocities,
            initial_guesses=self.initial_guesses,
            range_times=self.range_times[0],
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape, (3,))
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
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_direct_geocoding_monostatic_case0b(self) -> None:
        """Testing direct geocoding monostatic function, case 0b"""

        # case 0b: 1 pos (1,3), 1 vel (1,3), 1 rng time (float), 1 initial guess (1,3)
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions.reshape(1, 3),
            sensor_velocities=self.scaled_arf_velocities.reshape(1, 3),
            initial_guesses=self.initial_guesses.reshape(1, 3),
            range_times=self.range_times[0],
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
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
            out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case0c(self) -> None:
        """Testing direct geocoding monostatic function, case 0c"""

        # case 0c: 1 pos (3,), 1 vel (3,), 1 rng time (float), no initial guess
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions,
            sensor_velocities=self.scaled_arf_velocities,
            initial_guesses=None,
            range_times=self.range_times[0],
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape, (3,))
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
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_direct_geocoding_monostatic_case0d(self) -> None:
        """Testing direct geocoding monostatic function, case 0d"""

        # case 0d: 1 pos (1,3), 1 vel (1,3), 1 rng time (float), no initial guess
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions.reshape(1, 3),
            sensor_velocities=self.scaled_arf_velocities.reshape(1, 3),
            initial_guesses=None,
            range_times=self.range_times[0],
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
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
            out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case1a(self) -> None:
        """Testing direct geocoding monostatic function, case 1a"""

        # case 1a: N pos (N,3), N vel (N,3), 1 rng time (float), 1 initial guess (3,)
        out = direct_geocoding_monostatic(
            sensor_positions=np.full((self.N, 3), self.positions),
            sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
            initial_guesses=self.initial_guesses,
            range_times=self.range_times[0],
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
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
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case1b(self) -> None:
        """Testing direct geocoding monostatic function, case 1b"""

        # case 1b: N pos (N,3), N vel (N,3), 1 rng time (float), 1 initial guess (1,3)
        out = direct_geocoding_monostatic(
            sensor_positions=np.full((self.N, 3), self.positions),
            sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
            initial_guesses=self.initial_guesses.reshape(1, 3),
            range_times=self.range_times[0],
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
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
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case1c(self) -> None:
        """Testing direct geocoding monostatic function, case 1c"""

        # case 1c: N pos (N,3), N vel (N,3), 1 rng time (float), N initial guesses (N,3)
        out = direct_geocoding_monostatic(
            sensor_positions=np.full((self.N, 3), self.positions),
            sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
            initial_guesses=np.full((self.N, 3), self.initial_guesses),
            range_times=self.range_times[0],
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
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
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case1d(self) -> None:
        """Testing direct geocoding monostatic function, case 1d"""

        # case 1d: N pos (N,3), N vel (N,3), 1 rng time (float), no initial guess
        out = direct_geocoding_monostatic(
            sensor_positions=np.full((self.N, 3), self.positions),
            sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
            initial_guesses=None,
            range_times=self.range_times[0],
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
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
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case2a(self) -> None:
        """Testing direct geocoding monostatic function, case 2a"""

        # case 2a: 1 pos (3,), 1 vel (3,), M rng times (M,), 1 initial guess (3,)
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions,
            sensor_velocities=self.scaled_arf_velocities,
            initial_guesses=self.initial_guesses,
            range_times=np.repeat(self.range_times[0], self.M),
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
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
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case2b(self) -> None:
        """Testing direct geocoding monostatic function, case 2b"""

        # case 2b: 1 pos (1,3), 1 vel (1,3), M rng times (M,), 1 initial guess (1,3)
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions.reshape(1, 3),
            sensor_velocities=self.scaled_arf_velocities.reshape(1, 3),
            initial_guesses=self.initial_guesses.reshape(1, 3),
            range_times=np.repeat(self.range_times[0], self.M),
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
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
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case2c(self) -> None:
        """Testing direct geocoding monostatic function, case 2c"""

        # case 2c: 1 pos (3,), 1 vel (3,), M rng times (M,), no initial guess
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions,
            sensor_velocities=self.scaled_arf_velocities,
            initial_guesses=None,
            range_times=np.repeat(self.range_times[0], self.M),
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
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
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case2d(self) -> None:
        """Testing direct geocoding monostatic function, case 2d"""

        # case 2d: 1 pos (3,), 1 vel (3,), M rng times (M,), 1 initial guess (3,), M doppler freqs
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions,
            sensor_velocities=self.scaled_arf_velocities,
            initial_guesses=self.initial_guesses,
            range_times=np.repeat(self.range_times[0], self.M),
            frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
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
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case2e(self) -> None:
        """Testing direct geocoding monostatic function, case 2e"""

        # case 2e: 1 pos (1,3), 1 vel (1,3), M rng times (M,), 1 initial guess (1,3), M doppler freqs
        out = direct_geocoding_monostatic(
            sensor_positions=self.positions.reshape(1, 3),
            sensor_velocities=self.scaled_arf_velocities.reshape(1, 3),
            initial_guesses=self.initial_guesses.reshape(1, 3),
            range_times=np.repeat(self.range_times[0], self.M),
            frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
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
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case3a(self) -> None:
        """Testing direct geocoding monostatic function, case 3a"""

        # case 3a: N pos (N,3), N vel (N,3), M rng times (M, ), 1 initial guess (3,)
        out = direct_geocoding_monostatic(
            sensor_positions=np.full((self.N, 3), self.positions),
            sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
            initial_guesses=self.initial_guesses,
            range_times=np.repeat(self.range_times[0], self.M),
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out[0, ...],
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
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
            out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case3b(self) -> None:
        """Testing direct geocoding monostatic function, case 3b"""

        # case 3b: N pos (N,3), N vel (N,3), M rng times (M, ), no initial guess
        out = direct_geocoding_monostatic(
            sensor_positions=np.full((self.N, 3), self.positions),
            sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
            initial_guesses=None,
            range_times=np.repeat(self.range_times[0], self.M),
            frequencies_doppler_centroid=self.doppler_freqs,
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out[0, ...],
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
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
            out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case3c(self) -> None:
        """Testing direct geocoding monostatic function, case 3c"""

        # case 3c: N pos (N,3), N vel (N,3), M rng times (M, ), 1 initial guess (3,), M doppler freqs
        out = direct_geocoding_monostatic(
            sensor_positions=np.full((self.N, 3), self.positions),
            sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
            initial_guesses=self.initial_guesses,
            range_times=np.repeat(self.range_times[0], self.M),
            frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
            geodetic_altitude=self.geodetic_altitude,
            geocoding_side=self.look_direction,
            wavelength=self.wavelength,
        )

        doppler_residual = _doppler_equation_residual(
            sensor_pos=self.positions,
            sensor_vel=self.scaled_arf_velocities,
            ground_points=out[0, ...],
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual(
            sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
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
            out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_case4(self) -> None:
        """Testing direct geocoding monostatic function, case 4"""

        # case 4: N pos (N,3), M vel (M,3), raising error mismatch position/velocity
        with self.assertRaises(AmbiguousInputCorrelation):
            direct_geocoding_monostatic(
                sensor_positions=np.full((self.N, 3), self.positions),
                sensor_velocities=np.full((self.M, 3), self.scaled_arf_velocities),
                initial_guesses=np.full((self.N, 3), self.initial_guesses),
                range_times=np.repeat(self.range_times[0], self.M),
                frequencies_doppler_centroid=self.doppler_freqs,
                geodetic_altitude=self.geodetic_altitude,
                geocoding_side=self.look_direction,
                wavelength=self.wavelength,
            )

    def test_direct_geocoding_monostatic_case5(self) -> None:
        """Testing direct geocoding monostatic function, case 5"""

        # case 5: N pos (N,3), M init guesses (M,3), raising error mismath position/init guesses
        with self.assertRaises(AmbiguousInputCorrelation):
            direct_geocoding_monostatic(
                sensor_positions=np.full((self.N, 3), self.positions),
                sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
                initial_guesses=np.full((self.N // 2, 3), self.initial_guesses),
                range_times=np.repeat(self.range_times[0], self.M),
                frequencies_doppler_centroid=self.doppler_freqs,
                geodetic_altitude=self.geodetic_altitude,
                geocoding_side=self.look_direction,
                wavelength=self.wavelength,
            )

    def test_direct_geocoding_monostatic_case6(self) -> None:
        """Testing direct geocoding monostatic function, case 6"""

        # case 6: N range (N,), M freqs (M,), raising error mismath frequency/ranges
        with self.assertRaises(AmbiguousInputCorrelation):
            direct_geocoding_monostatic(
                sensor_positions=np.full((self.N, 3), self.positions),
                sensor_velocities=np.full((self.N, 3), self.scaled_arf_velocities),
                initial_guesses=np.full((self.N, 3), self.initial_guesses),
                range_times=np.repeat(self.range_times[0], self.N),
                frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
                geodetic_altitude=self.geodetic_altitude,
                geocoding_side=self.look_direction,
                wavelength=self.wavelength,
            )


class DirectGeocodingMonostaticCoreTest(unittest.TestCase):
    """Testing direct geocoding monostatic core"""

    def setUp(self):
        """Setting up variables for testing"""
        self.position = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004],
        )
        self.velocity = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346],
        )
        self.initial_guess = np.array(
            [4385882.165360568, 764600.914414172, 4551967.490551733]
        )
        self.range_time = np.array([2.05624579e-05])
        self.doppler_freq = 0
        self.geodetic_altitude = 0
        self.wavelength = 1

        self.N = 4
        self.M = 5
        self.tolerance = 1e-5

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def test_monostatic_core_0a(self) -> None:
        """Test _geocoding_monostatic_core function, case 0a"""

        # case 0a: 1 pos (3,), 1 vel (3,), 1 guess (3,), 1 rng time
        out = _direct_geocoding_monostatic_core(
            sensor_positions=self.position,
            sensor_velocities=self.velocity,
            initial_guesses=self.initial_guess,
            range_times=self.range_time[0],
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_monostatic_core_0b(self) -> None:
        """Test _geocoding_monostatic_core function, case 0b"""

        # case 0b: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), 1 rng time
        out = _direct_geocoding_monostatic_core(
            sensor_positions=self.position.reshape(1, 3),
            sensor_velocities=self.velocity.reshape(1, 3),
            initial_guesses=self.initial_guess.reshape(1, 3),
            range_times=self.range_time[0],
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
        np.testing.assert_allclose(
            out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_monostatic_core_0c(self) -> None:
        """Test _geocoding_monostatic_core function, case 0c"""

        # case 0c: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times
        out = _direct_geocoding_monostatic_core(
            sensor_positions=self.position,
            sensor_velocities=self.velocity,
            initial_guesses=self.initial_guess,
            range_times=np.repeat(self.range_time[0], self.M),
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
        np.testing.assert_allclose(
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_monostatic_core_0d(self) -> None:
        """Test _geocoding_monostatic_core function, case 0d"""

        # case 0d: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times
        out = _direct_geocoding_monostatic_core(
            sensor_positions=self.position.reshape(1, 3),
            sensor_velocities=self.velocity.reshape(1, 3),
            initial_guesses=self.initial_guess.reshape(1, 3),
            range_times=np.repeat(self.range_time[0], self.M),
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
        np.testing.assert_allclose(
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_monostatic_core_0e(self) -> None:
        """Test _geocoding_monostatic_core function, case 0e"""

        # case 0e: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times, M doppler freqs
        out = _direct_geocoding_monostatic_core(
            sensor_positions=self.position,
            sensor_velocities=self.velocity,
            initial_guesses=self.initial_guess,
            range_times=np.repeat(self.range_time[0], self.M),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
        np.testing.assert_allclose(
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_monostatic_core_0f(self) -> None:
        """Test _geocoding_monostatic_core function, case 0f"""

        # case 0f: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times, M doppler freqs
        out = _direct_geocoding_monostatic_core(
            sensor_positions=self.position.reshape(1, 3),
            sensor_velocities=self.velocity.reshape(1, 3),
            initial_guesses=self.initial_guess.reshape(1, 3),
            range_times=np.repeat(self.range_time[0], self.M),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
        np.testing.assert_allclose(
            out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_monostatic_core_1a(self) -> None:
        """Test _geocoding_monostatic_core function, case 1a"""

        # case 1a: 1 pos (N,3), 1 vel (N,3), 1 guess (N,3), 1 rng time
        out = _direct_geocoding_monostatic_core(
            sensor_positions=np.full((self.N, 3), self.position),
            sensor_velocities=np.full((self.N, 3), self.velocity),
            initial_guesses=np.full((self.N, 3), self.initial_guess),
            range_times=self.range_time[0],
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
        np.testing.assert_allclose(
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_monostatic_core_1b(self) -> None:
        """Test _geocoding_monostatic_core function, case 1b"""

        # case 1b: 1 pos (N,3), 1 vel (N,3), 1 guess (N,3), M rng times
        out = _direct_geocoding_monostatic_core(
            sensor_positions=np.full((self.N, 3), self.position),
            sensor_velocities=np.full((self.N, 3), self.velocity),
            initial_guesses=np.full((self.N, 3), self.initial_guess),
            range_times=np.repeat(self.range_time[0], self.M),
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(
            out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_monostatic_core_1c(self) -> None:
        """Test _geocoding_monostatic_core function, case 1c"""

        # case 1c: N pos (N,3), N vel (N,3), N guesses (N,3), M rng times, M doppler freqs
        out = _direct_geocoding_monostatic_core(
            sensor_positions=np.full((self.N, 3), self.position),
            sensor_velocities=np.full((self.N, 3), self.velocity),
            initial_guesses=np.full((self.N, 3), self.initial_guess),
            range_times=np.repeat(self.range_time[0], self.M),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(
            out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_monostatic_core_2(self) -> None:
        """Test _geocoding_monostatic_core function, case 2"""

        # case 2: N pos (N,3), N vel (N,3), N guesses (N,3), M rng times, N doppler freqs
        # assert error raising
        with self.assertRaises(AmbiguousInputCorrelation):
            _direct_geocoding_monostatic_core(
                sensor_positions=np.full((self.N, 3), self.position),
                sensor_velocities=np.full((self.N, 3), self.velocity),
                initial_guesses=np.full((self.N, 3), self.initial_guess),
                range_times=np.repeat(self.range_time[0], self.M),
                frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
                geodetic_altitude=self.geodetic_altitude,
            )


class NewtonForDirectGeocodingMonostaticTest(unittest.TestCase):
    """Testing Newton method for direct geocoding vectorized"""

    def setUp(self):
        self.position = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004]
        )
        self.velocity = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346]
        )
        self.init_guess = np.array(
            [4385932.628762595, 764443.4718341012, 4551945.624046889]
        )
        self.geodetic_altitude = 0
        self.wavelength = 1
        self.doppler_frequency = 0
        self.range_time = 2.05624579e-05

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])
        self.tolerance = 1e-5

    def test_newton_for_geocoding_array_case0a(self) -> None:
        """Testing Newton for geocoding for array input, case 0a"""
        out = _newton_for_direct_geocoding_monostatic(
            sensor_positions=self.position,
            sensor_velocities=self.velocity,
            initial_guesses=self.init_guess,
            range_time=self.range_time,
            geodetic_altitude=self.geodetic_altitude,
            wavelength=self.wavelength,
            frequency_doppler_centroid=self.doppler_frequency,
        )

        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_newton_for_geocoding_array_case0b(self) -> None:
        """Testing Newton for geocoding for array input, case 0b"""
        out = _newton_for_direct_geocoding_monostatic(
            sensor_positions=self.position.reshape(1, 3),
            sensor_velocities=self.velocity.reshape(1, 3),
            initial_guesses=self.init_guess.reshape(1, 3),
            range_time=self.range_time,
            geodetic_altitude=self.geodetic_altitude,
            wavelength=self.wavelength,
            frequency_doppler_centroid=self.doppler_frequency,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
        np.testing.assert_allclose(
            out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_newton_for_geocoding_array_case1(self) -> None:
        """Testing Newton for geocoding for array input, case 0b"""
        out = _newton_for_direct_geocoding_monostatic(
            sensor_positions=np.full((4, 3), self.position),
            sensor_velocities=np.full((4, 3), self.velocity),
            initial_guesses=np.full((4, 3), self.init_guess),
            range_time=self.range_time,
            geodetic_altitude=self.geodetic_altitude,
            wavelength=self.wavelength,
            frequency_doppler_centroid=self.doppler_frequency,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_allclose(
            out, np.full((4, 3), self.results), atol=self.tolerance, rtol=0
        )


class DirectGeocodingMonostaticInitTest(unittest.TestCase):
    """Testing direct_geocoding_monostatic_init"""

    def setUp(self):
        self.position = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004]
        )
        self.velocity = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346]
        )
        self.init_guess = np.array(
            [4385932.628762595, 764443.4718341012, 4551945.624046889]
        )
        self.geodetic_altitude = 0
        self.wavelength = 1
        self.doppler_frequency = 0
        range_time = 2.05624579e-05
        self.look_direction = GeocodingSide.RIGHT_LOOKING
        self.range_distance = np.median(range_time) * speed_of_light / 2

        self.N = 4
        self.results = np.array(
            [4385882.165360568, 764600.914414172, 4551967.490551733]
        )
        self.tolerance = 1e-5

    def test_direct_geocoding_monostatic_init_case0a(self) -> None:
        """Testing direct geocoding monostatic init, case 0a"""

        # case0a: 1 sensor pos (3,), 1 sensor vel (3,)
        out = direct_geocoding_monostatic_init(
            sensor_positions=self.position,
            sensor_velocities=self.velocity,
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 1)
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_direct_geocoding_monostatic_init_case0b(self) -> None:
        """Testing direct geocoding monostatic init, case 0b"""

        # case0b: 1 sensor pos (1, 3), 1 sensor vel (3,)
        out = direct_geocoding_monostatic_init(
            sensor_positions=self.position.reshape(1, 3),
            sensor_velocities=self.velocity,
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(
            out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_init_case0c(self) -> None:
        """Testing direct geocoding monostatic init, case 0c"""

        # case0c: 1 sensor pos (3,), 1 sensor vel (1, 3)
        out = direct_geocoding_monostatic_init(
            sensor_positions=self.position,
            sensor_velocities=self.velocity.reshape(1, 3),
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(
            out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_init_case0d(self) -> None:
        """Testing direct geocoding monostatic init, case 0d"""

        # case0d: 1 sensor pos (1, 3), 1 sensor vel (1, 3)
        out = direct_geocoding_monostatic_init(
            sensor_positions=self.position.reshape(1, 3),
            sensor_velocities=self.velocity.reshape(1, 3),
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(
            out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_init_case1a(self) -> None:
        """Testing direct geocoding monostatic init, case 1a"""

        # case1a: N sensor pos (N, 3), 1 sensor vel
        out = direct_geocoding_monostatic_init(
            sensor_positions=np.full((self.N, 3), self.position),
            sensor_velocities=self.velocity,
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_init_case1b(self) -> None:
        """Testing direct geocoding monostatic init, case 1b"""

        # case1b: N sensor pos (N, 3), 1 sensor vel (1, 3)
        out = direct_geocoding_monostatic_init(
            sensor_positions=np.full((self.N, 3), self.position),
            sensor_velocities=self.velocity.reshape(1, 3),
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_init_case1c(self) -> None:
        """Testing direct geocoding monostatic init, case 1c"""

        # case1c: 1 sensor pos (3,), N sensor vel (N, 3)
        out = direct_geocoding_monostatic_init(
            sensor_positions=self.position,
            sensor_velocities=np.full((self.N, 3), self.velocity),
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_init_case1d(self) -> None:
        """Testing direct geocoding monostatic init, case 1d"""

        # case1d: 1 sensor pos (1, 3), N sensor vel (N, 3)
        out = direct_geocoding_monostatic_init(
            sensor_positions=self.position.reshape(1, 3),
            sensor_velocities=np.full((self.N, 3), self.velocity),
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_monostatic_init_case1e(self) -> None:
        """Testing direct geocoding monostatic init, case 1e"""

        # case1e: 1 sensor pos (1, 3), N sensor vel (N, 3)
        out = direct_geocoding_monostatic_init(
            sensor_positions=np.full((self.N, 3), self.position),
            sensor_velocities=np.full((self.N, 3), self.velocity),
            range_distance=self.range_distance,
            geocoding_side=self.look_direction,
        )
        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(
            out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
