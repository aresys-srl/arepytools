# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

from arepytools.geometry.conversions import llh2xyz
from arepytools.geometry.generalsarorbit import (
    GeneralSarOrbit,
    compute_anx_times,
    compute_ground_velocity,
    compute_incidence_angles_from_orbit,
    compute_look_angles_from_orbit,
    compute_number_of_anx,
    create_general_sar_orbit,
)
from arepytools.io.metadata import StateVectors
from arepytools.math.axis import RegularAxis
from arepytools.timing.precisedatetime import PreciseDateTime


def _identify_anx_time_intervals(time_axis, state_vectors):
    number_of_state_vectors = state_vectors.shape[0]
    number_of_state_vector_intervals = number_of_state_vectors - 1

    anx_time_intervals = []

    for interval_index in range(number_of_state_vector_intervals):
        interval_begin_index, interval_end_index = interval_index, interval_index + 2
        z_start, z_stop = state_vectors[interval_begin_index:interval_end_index, 2]

        if z_start <= 0 < z_stop:
            t_start, t_stop = time_axis[interval_begin_index:interval_end_index]
            anx_time_intervals.append((t_start, t_stop))

    return anx_time_intervals


class GeneralSarOrbitTestCase(unittest.TestCase):
    def setUp(self):
        height = 800000
        self._state_vectors = np.array(
            [
                llh2xyz([-1e-4, 0, height]),
                llh2xyz([5e-5, 1e-4, height]),
                llh2xyz([2e-4, 2e-4, height]),
                llh2xyz([1e-4, 3e-4, height]),
                llh2xyz([0, 4e-4, height]),
                llh2xyz([-1e-4, 5e-4, height]),
                llh2xyz([0, 6e-4, height]),
                llh2xyz([1e-4, 7e-4, height]),
                llh2xyz([5e-5, 8e-4, height]),
                llh2xyz([-5e-5, 9e-4, height]),
            ]
        ).squeeze()

        number_of_state_vectors = self._state_vectors.shape[0]

        time_axis_step = 1
        self._time_axis = np.array(
            [
                PreciseDateTime(state_vector_index * time_axis_step)
                for state_vector_index in range(number_of_state_vectors)
            ]
        )

    def test_compute_anx__generic(self):
        orbit = GeneralSarOrbit(self._time_axis, self._state_vectors.ravel())

        anx_time_intervals = _identify_anx_time_intervals(
            self._time_axis, self._state_vectors
        )
        number_of_anx = len(anx_time_intervals)

        max_abs_z_error = 1e-3
        anx_times = compute_anx_times(orbit, max_abs_z_error=max_abs_z_error)

        anx_positions_from_times = orbit.get_position(anx_times)
        anx_velocities_from_times = orbit.get_velocity(anx_times)

        self.assertEqual(anx_times.ndim, 1)

        self.assertEqual(anx_times.size, number_of_anx)

        self.assertTrue(
            all(
                np.abs(anx_pos_z) <= max_abs_z_error
                for anx_pos_z in anx_positions_from_times[2, :]
            )
        )
        self.assertTrue(
            all(anx_vel_z >= 0 for anx_vel_z in anx_velocities_from_times[2, :])
        )

        for anx_time_interval in anx_time_intervals:
            anx_found = False
            for anx_time in anx_times:
                if anx_time_interval[0] <= anx_time <= anx_time_interval[1]:
                    anx_found = True
                    break

            self.assertTrue(anx_found)

    def test_compute_number_of_anx__generic(self):
        anx_time_intervals = _identify_anx_time_intervals(
            self._time_axis, self._state_vectors
        )
        number_of_anx_ref = len(anx_time_intervals)

        orbit = GeneralSarOrbit(self._time_axis, self._state_vectors.ravel())

        number_of_anx = compute_number_of_anx(orbit)

        self.assertEqual(number_of_anx, number_of_anx_ref)

    def test_constructor__none_anx_times(self):
        orbit = GeneralSarOrbit(
            self._time_axis, self._state_vectors.ravel(), anx_times_evaluator=None
        )
        self.assertIsNone(orbit.anx_times)
        self.assertIsNone(orbit.anx_positions)

    def test_get_time_since_anx__generic(self):
        orbit = GeneralSarOrbit(self._time_axis, self._state_vectors.ravel())

        self.assertIsNotNone(orbit.anx_times)
        assert orbit.anx_times is not None

        absolute_times = self._time_axis
        relative_times, anx_time_indices = orbit.get_time_since_anx(absolute_times)

        self.assertIsNotNone(relative_times)
        assert relative_times is not None
        self.assertIsNotNone(anx_time_indices)
        assert anx_time_indices is not None

        self.assertEqual(len(absolute_times), len(relative_times))
        self.assertEqual(len(relative_times), len(anx_time_indices))

        for absolute_time, relative_time, anx_time_index in zip(
            absolute_times, relative_times, anx_time_indices
        ):
            if np.isnan(relative_time):
                self.assertLess(absolute_time, orbit.anx_times[0])
                with self.assertRaises(IndexError):
                    orbit.anx_times[anx_time_index]
            else:
                self.assertGreaterEqual(relative_time, 0.0)
                for anx_time in orbit.anx_times:
                    self.assertTrue(
                        absolute_time - anx_time < 0.0
                        or relative_time <= absolute_time - anx_time
                    )

                self.assertEqual(
                    absolute_time - orbit.anx_times[anx_time_index], relative_time
                )


class CreateGeneralSarOrbitTestCase(unittest.TestCase):
    def setUp(self):
        npoints = GeneralSarOrbit.get_minimum_number_of_data_points()
        position_vector = np.tile([1.0, 2.0, 3.0], (1, npoints)).T
        velocity_vector = np.zeros_like(position_vector)

        t_ref_utc = PreciseDateTime(10.0, 0.0)
        dt_sv_s = 1.0

        self.state_vectors = StateVectors(
            position_vector, velocity_vector, t_ref_utc=t_ref_utc, dt_sv_s=dt_sv_s
        )

        self.anx_time = PreciseDateTime(0.0, 0.0)
        self.invalid_anx_time = PreciseDateTime(11.0, 0.0)
        self.anx_position = [1.0, 2.0, -3.0]

    def _assert_anx_info_is_none(self, orbit: GeneralSarOrbit):
        self.assertIsNotNone(orbit.anx_times)
        assert orbit.anx_times is not None
        np.testing.assert_array_equal(orbit.anx_times, np.empty((0,)))

        self.assertIsNotNone(orbit.anx_positions)
        assert orbit.anx_positions is not None
        np.testing.assert_array_equal(orbit.anx_positions, np.empty((3, 0)))

    def _assert_anx_info_is_not_none(
        self,
        orbit: GeneralSarOrbit,
    ):
        self.assertIsNotNone(orbit.anx_times)
        assert orbit.anx_times is not None
        np.testing.assert_array_equal(orbit.anx_times, np.array([self.anx_time]))

        self.assertIsNotNone(orbit.anx_positions)
        assert orbit.anx_positions is not None
        np.testing.assert_array_equal(
            orbit.anx_positions, np.array([self.anx_position]).reshape((3, 1))
        )

    def test_create_general_sarorbit(self):
        self.state_vectors.set_anx_info(self.anx_time, self.anx_position)

        orbit = create_general_sar_orbit(self.state_vectors)
        self._assert_anx_info_is_not_none(orbit)

        orbit = create_general_sar_orbit(self.state_vectors, True)
        self._assert_anx_info_is_not_none(orbit)

        orbit = create_general_sar_orbit(self.state_vectors, False)
        self._assert_anx_info_is_not_none(orbit)

    def test_create_general_sarorbit_invalid_anx(self):
        self.state_vectors.set_anx_info(self.invalid_anx_time, self.anx_position)

        with self.assertRaises(ValueError):
            orbit = create_general_sar_orbit(self.state_vectors)

        orbit = create_general_sar_orbit(self.state_vectors, True)
        self._assert_anx_info_is_none(orbit)

        with self.assertRaises(ValueError):
            orbit = create_general_sar_orbit(self.state_vectors, False)

    def test_create_general_sarorbit_unspecified_anx(self):
        orbit = create_general_sar_orbit(self.state_vectors)
        self._assert_anx_info_is_none(orbit)

        orbit = create_general_sar_orbit(self.state_vectors, True)
        self._assert_anx_info_is_none(orbit)

        orbit = create_general_sar_orbit(self.state_vectors, False)
        self._assert_anx_info_is_none(orbit)


class GeometryFunctionHelpersTestCase(unittest.TestCase):
    _state_vectors = np.array(
        [
            5317606.94350283,
            610603.985945038,
            4577936.89859885,
            5313024.53547427,
            608285.563877273,
            4583547.15708167,
            5308435.7651548,
            605967.120830312,
            4589152.18047604,
            5303840.63790599,
            603648.660435838,
            4594751.96221552,
            5299239.15894225,
            601330.18624638,
            4600346.49592944,
            5294631.33350784,
            599011.701824865,
            4605935.7752263,
            5290017.16682646,
            596693.210719223,
            4611519.79375494,
        ]
    )

    def setUp(self):
        num_of_state_vectors = self._state_vectors.size // 3

        origin = PreciseDateTime.from_utc_string("01-JAN-2021 00:00:00.000000000000")
        time_axis = RegularAxis((0, 1, num_of_state_vectors), origin)

        self.orbit = GeneralSarOrbit(time_axis, self._state_vectors)

    def test_compute_ground_velocity(self):
        time_point = self.orbit.t0

        look_angles = np.deg2rad(np.arange(15.0, 50.0, 5.0))

        reference_ground_velocity = np.array(
            [
                6858.345099538182,
                6849.722806175047,
                6839.602711361663,
                6827.447486757614,
                6812.427011180386,
                6793.187664343439,
                6767.353714800150,
            ]
        )

        computed_ground_velocity = compute_ground_velocity(
            self.orbit, time_point, look_angles
        )

        self.assertEqual(np.shape(computed_ground_velocity), look_angles.shape)
        np.testing.assert_allclose(computed_ground_velocity, reference_ground_velocity)

        computed_ground_velocity_0 = compute_ground_velocity(
            self.orbit, time_point, look_angles[0]
        )

        self.assertIsInstance(computed_ground_velocity_0, float)
        np.testing.assert_allclose(
            computed_ground_velocity_0, reference_ground_velocity[0]
        )

    def test_compute_look_angles_from_orbit(self):
        time_point = self.orbit.t0

        range_times = np.array(
            [
                0.004673992827008,
                0.004820153606007,
                0.005020843688030,
                0.005288165093247,
                0.005640692134166,
                0.006107790154196,
                0.006738512242025,
            ]
        )

        reference = np.deg2rad(
            np.array(
                [
                    15.000477295142909,
                    20.000405360157288,
                    25.000361305240148,
                    30.000331169663703,
                    35.000308965284916,
                    40.000291693285284,
                    45.000277682902080,
                ]
            )
        )

        look_angles = compute_look_angles_from_orbit(
            self.orbit, time_point, range_times, "RIGHT"
        )
        self.assertEqual(np.shape(look_angles), range_times.shape)
        np.testing.assert_allclose(look_angles, reference)

        look_angles = compute_look_angles_from_orbit(
            self.orbit, time_point, range_times[0], "RIGHT"
        )
        self.assertIsInstance(look_angles, float)
        np.testing.assert_allclose(look_angles, reference[0])

        look_angles = compute_look_angles_from_orbit(
            self.orbit,
            time_point,
            range_times[0],
            "RIGHT",
            doppler_centroid=0.0,
            carrier_wavelength=1.0,
        )
        self.assertIsInstance(look_angles, float)
        np.testing.assert_allclose(look_angles, reference[0])

        look_angles = compute_look_angles_from_orbit(
            self.orbit,
            time_point,
            range_times,
            "RIGHT",
            doppler_centroid=0.0,
            carrier_wavelength=1.0,
        )
        self.assertEqual(np.shape(look_angles), range_times.shape)
        np.testing.assert_allclose(look_angles, reference)

        look_angles = compute_look_angles_from_orbit(
            self.orbit,
            time_point,
            range_times,
            "RIGHT",
            doppler_centroid=np.zeros_like(range_times),
            carrier_wavelength=1.0,
        )
        self.assertEqual(np.shape(look_angles), range_times.shape)
        np.testing.assert_allclose(look_angles, reference)

    def test_compute_incidence_angles_from_orbit(self):
        time_point = self.orbit.t0

        range_times = np.array(
            [
                0.004673992827008,
                0.004820153606007,
                0.005020843688030,
                0.005288165093247,
                0.005640692134166,
                0.006107790154196,
                0.006738512242025,
            ]
        )

        reference = np.deg2rad(
            np.array(
                [
                    16.587391864029414,
                    22.179486058996705,
                    27.818340417194918,
                    33.523593567026595,
                    39.322848344314153,
                    45.257507178236310,
                    51.394652568444236,
                ]
            )
        )

        incidence_angles = compute_incidence_angles_from_orbit(
            self.orbit, time_point, range_times, "RIGHT"
        )
        self.assertEqual(np.shape(incidence_angles), range_times.shape)
        np.testing.assert_allclose(incidence_angles, reference)

        incidence_angles = compute_incidence_angles_from_orbit(
            self.orbit, time_point, range_times[0], "RIGHT"
        )
        self.assertIsInstance(incidence_angles, float)
        np.testing.assert_allclose(incidence_angles, reference[0])

        incidence_angles = compute_incidence_angles_from_orbit(
            self.orbit,
            time_point,
            range_times[0],
            "RIGHT",
            doppler_centroid=0.0,
            carrier_wavelength=1.0,
        )
        self.assertIsInstance(incidence_angles, float)
        np.testing.assert_allclose(incidence_angles, reference[0])

        incidence_angles = compute_incidence_angles_from_orbit(
            self.orbit,
            time_point,
            range_times,
            "RIGHT",
            doppler_centroid=0.0,
            carrier_wavelength=1.0,
        )
        self.assertEqual(np.shape(incidence_angles), range_times.shape)
        np.testing.assert_allclose(incidence_angles, reference)

        incidence_angles = compute_incidence_angles_from_orbit(
            self.orbit,
            time_point,
            range_times,
            "RIGHT",
            doppler_centroid=np.zeros_like(range_times),
            carrier_wavelength=1.0,
        )
        self.assertEqual(np.shape(incidence_angles), range_times.shape)
        np.testing.assert_allclose(incidence_angles, reference)


class GeneralSAROrbitFullOrbitTestCase(unittest.TestCase):
    _state_vectors = np.array(
        [
            -779001.28129911,
            -505870.7546934,
            6817987.21562447,
            -1787813.04699317,
            629706.49654234,
            6614777.74861071,
            -2676777.49486916,
            1762403.33562427,
            6089174.4622466,
            -3403702.99091361,
            2833140.93065613,
            5266805.03570274,
            -3935963.30795762,
            3784762.95672522,
            4187763.77972863,
            -4251980.19525865,
            4565112.00564628,
            2904652.08879896,
            -4342081.19344017,
            5129895.41407931,
            1480008.94949346,
            -4208699.84360167,
            5445182.53365344,
            -16741.96566374,
            -3865922.04471967,
            5489393.41476849,
            -1512678.67406476,
            -3338417.95554259,
            5254664.63265023,
            -2934934.90842229,
            -2659831.54067513,
            4747510.15094551,
            -4214248.33821238,
            -1870727.8761017,
            3988731.95503892,
            -5288330.41117336,
            -1016220.22596924,
            3012574.72049191,
            -6104894.47001853,
            -143413.63392233,
            1865158.91592616,
            -6624195.83061395,
            701191.28183798,
            602265.36994385,
            -6820961.43577,
            1474191.68582983,
            -713420.57171269,
            -6685616.32885961,
            2137539.09251902,
            -2015467.38227689,
            -6224748.10095948,
            2660441.17922355,
            -3237097.45252941,
            -5460787.06866317,
            3020812.70959749,
            -4314653.48632447,
            -4430917.553466,
            3206206.3232972,
            -5190950.00548857,
            -3185272.53864321,
            3214184.02923558,
            -5818332.44350126,
            -1784498.48374172,
            3052122.01979767,
            -6161279.32465417,
            -296807.57728266,
            2736473.1183581,
            -6198404.88017072,
            1205340.24567249,
            2291541.0597032,
            -5923749.16632974,
            2648764.83733911,
            1747847.25026194,
            -5347278.90699311,
            3963131.52602845,
            1140192.21257602,
            -4494563.084553,
            5084381.11411162,
            505529.40108162,
            -3405630.64317499,
            5957856.38589271,
            -119222.36344083,
            -2133061.23760086,
            6540971.68296547,
            -699300.40803076,
            -739401.37358005,
            6805294.15498011,
        ]
    )

    def setUp(self):
        num_of_state_vectors = self._state_vectors.size // 3
        origin = PreciseDateTime.from_utc_string("27-JUL-2021 23:28:48.000005006790")
        time_axis = RegularAxis((0, 200, num_of_state_vectors), origin)
        self.orbit = GeneralSarOrbit(time_axis, self._state_vectors.ravel())
        self.reference_azimuth_time = PreciseDateTime.from_utc_string(
            "28-JUL-2021 00:51:30.187206689901"
        )
        self.reference_range_time = 0.003772314315288605
        self.point = np.array([1366383.87575117, -4229928.08430407, 4558648.89297629])

    def test_bistatic_inverse_geocoding(self):
        az, rg = self.orbit.earth2sat(self.point, orbit_tx=self.orbit)

        self.assertEqual(len(az), 1)
        self.assertEqual(len(rg), 1)
        self.assertAlmostEqual(az[0] - self.reference_azimuth_time, 0.0)
        self.assertAlmostEqual(rg[0] - self.reference_range_time, 0.0)

    def test_monostatic_sat2earth_single_range(self):
        point = self.orbit.sat2earth(
            self.reference_azimuth_time, self.reference_range_time, "RIGHT"
        )

        ref_point = np.array(
            [1366378.3848378758, -4229919.984669094, 4558657.993018785]
        )
        self.assertIsInstance(point, np.ndarray)
        self.assertEqual(point.shape, (3, 1))
        self.assertEqual(point.ndim, 2)
        self.assertEqual(point.size, 3)
        np.testing.assert_allclose(point, ref_point.reshape(-1, 1), atol=1e-8, rtol=0)

    def test_monostatic_sat2earth_multiple_ranges(self):
        rip = 5
        point = self.orbit.sat2earth(
            self.reference_azimuth_time, [self.reference_range_time] * rip, "RIGHT"
        )

        ref_point = np.array(
            [1366378.3848378758, -4229919.984669094, 4558657.993018785]
        )
        self.assertIsInstance(point, np.ndarray)
        self.assertEqual(point.shape, (3, rip))
        self.assertEqual(point.ndim, 2)
        self.assertEqual(point.size, 3 * rip)
        np.testing.assert_allclose(
            point, np.full((rip, 3), ref_point).T, atol=1e-8, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
