# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import itertools
import unittest
from typing import List, Union

import numpy as np

from arepytools.geometry.generalsarattitude import (
    GeneralSarAttitude,
    compute_pointing_directions,
    create_general_sar_attitude,
)
from arepytools.geometry.generalsarorbit import GeneralSarOrbit
from arepytools.geometry.reference_frames import ReferenceFrame, RotationOrder
from arepytools.io.metadata import AttitudeInfo, StateVectors
from arepytools.math.axis import RegularAxis
from arepytools.timing.precisedatetime import PreciseDateTime


def _compute_antenna_angles_a_posteriori(antenna_reference_frame, vectors):
    vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    local_components = np.einsum("...jk, ...j->...k", antenna_reference_frame, vectors)
    azimuth_angles = np.arctan(local_components[..., 0] / local_components[..., 2])
    elevation_angles = np.arctan(local_components[..., 1] / local_components[..., 2])
    return azimuth_angles, elevation_angles


class GeneralSarAttitudeTestCase(unittest.TestCase):
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

    _yaw = np.array(
        [
            6.01558980504364e-06,
            1.72212173549011e-05,
            2.15175381458397e-05,
            1.89119065338883e-05,
            9.41149065433075e-06,
            5.92311556040909e-06,
            1.70738023389626e-05,
        ]
    )
    _pitch = np.array(
        [
            2.76942236127373e-06,
            7.98298153997112e-06,
            9.98141065677743e-06,
            8.76586934113811e-06,
            4.33952823916987e-06,
            2.72025187601975e-06,
            7.93086688889418e-06,
        ]
    )
    _roll = np.array(
        [
            25.9269737687551,
            25.9265769603434,
            25.9261802440172,
            25.9257836127292,
            25.9253870594373,
            25.9249872414186,
            25.9245819474607,
        ]
    )

    def setUp(self):
        self.time_axis = RegularAxis(
            (0, 1, self._state_vectors.size // 3),
            PreciseDateTime.from_utc_string("01-JAN-2021 00:00:00.000000000000"),
        )
        self.orbit = GeneralSarOrbit(self.time_axis, self._state_vectors)
        self.ypr_matrix = np.vstack((self._yaw, self._pitch, self._roll))

    def test_constructor(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )

        self.assertEqual(attitude.t0, self.time_axis.origin)
        self.assertEqual(attitude.dt, self.time_axis.step)
        self.assertEqual(attitude.n, self.time_axis.size)
        self.assertEqual(attitude.reference_frame, ReferenceFrame.zero_doppler)
        self.assertEqual(attitude.rotation_order, RotationOrder.ypr)
        np.testing.assert_equal(attitude.ypr_angles, self.ypr_matrix)
        np.testing.assert_equal(attitude.time_axis_array, self.time_axis.get_array())

    def test_constructor_invalid_ypr_shape(self):
        with self.assertRaises(RuntimeError):
            GeneralSarAttitude(
                self.orbit, self.time_axis, self.ypr_matrix.T, "YPR", "ZERODOPPLER"
            )
        with self.assertRaises(RuntimeError):
            GeneralSarAttitude(
                self.orbit,
                self.time_axis,
                self.ypr_matrix[:, 0:2],
                "YPR",
                "ZERODOPPLER",
            )
        with self.assertRaises(RuntimeError):
            GeneralSarAttitude(
                self.orbit,
                self.time_axis,
                self.ypr_matrix[:, 0:6],
                "YPR",
                "ZERODOPPLER",
            )

    def test_non_regular_axis(self):
        attitude = GeneralSarAttitude(
            self.orbit,
            self.time_axis.get_array(),
            self.ypr_matrix,
            "YPR",
            "ZERODOPPLER",
        )

        self.assertEqual(attitude.t0, self.time_axis.origin)
        with self.assertRaises(RuntimeError):
            self.assertEqual(attitude.dt, self.time_axis.step)
        self.assertEqual(attitude.n, self.time_axis.size)
        self.assertEqual(attitude.reference_frame, ReferenceFrame.zero_doppler)
        self.assertEqual(attitude.rotation_order, RotationOrder.ypr)
        np.testing.assert_equal(attitude.ypr_angles, self.ypr_matrix)
        np.testing.assert_equal(attitude.time_axis_array, self.time_axis.get_array())

    def test_create_generalsarattitude(self):
        npoints = GeneralSarOrbit.get_minimum_number_of_data_points()
        position_vector = np.tile([1.0, 2.0, 3.0], (1, npoints)).T
        velocity_vector = np.zeros_like(position_vector)

        t_ref_utc = PreciseDateTime(10.0, 0.0)
        dt_sv_s = 1.0

        anx_time = PreciseDateTime(0.0, 0.0)
        anx_position = [1.0, 2.0, -3.0]

        state_vectors = StateVectors(
            position_vector, velocity_vector, t_ref_utc=t_ref_utc, dt_sv_s=dt_sv_s
        )
        state_vectors.set_anx_info(anx_time, anx_position)

        yaw = np.arange(npoints, dtype=float)
        pitch = np.arange(npoints, dtype=float)
        roll = np.arange(npoints, dtype=float)
        attitude_info = AttitudeInfo(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            t0=t_ref_utc,
            delta_t=dt_sv_s,
            ref_frame="ZERODOPPLER",
            rot_order="YPR",
        )

        attitude = create_general_sar_attitude(state_vectors, attitude_info)
        self.assertEqual(attitude.t0, t_ref_utc)
        self.assertEqual(attitude.dt, dt_sv_s)
        self.assertEqual(attitude.n, npoints)
        self.assertEqual(attitude.reference_frame, ReferenceFrame.zero_doppler)
        self.assertEqual(attitude.rotation_order, RotationOrder.ypr)
        np.testing.assert_equal(attitude.ypr_angles, np.vstack([yaw, pitch, roll]))
        np.testing.assert_equal(
            attitude.time_axis_array, np.arange(npoints) * dt_sv_s + t_ref_utc
        )

    def test_str(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )
        str(attitude)

    def test_get_arf(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )
        time_point = self.time_axis.origin + 3
        time_inputs = [time_point, np.asarray(self.time_axis.origin + 3).reshape((1,))]

        reference = np.array(
            [
                [-0.604580222175426, -0.564852727629684, -0.561626344684451],
                [-0.304829433093801, 0.815474732550541, -0.492016236816769],
                [0.735908806628936, -0.126263045507894, -0.665203631728697],
            ]
        )

        for time in time_inputs:
            arf = attitude.get_arf(time)

            self.assertEqual(arf.shape, np.shape(time) + (3, 3))
            np.testing.assert_allclose(
                arf, reference.reshape(arf.shape), rtol=1e-10, atol=1e-10
            )

    def test_get_arf_vectorized(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )
        time_point = self.time_axis.origin + 3
        arf = attitude.get_arf(np.tile(time_point, (3,)))

        reference = np.array(
            [
                [-0.604580222175426, -0.564852727629684, -0.561626344684451],
                [-0.304829433093801, 0.815474732550541, -0.492016236816769],
                [0.735908806628936, -0.126263045507894, -0.665203631728697],
            ]
        )

        np.testing.assert_allclose(
            arf, np.tile(reference, (3, 1, 1)), rtol=1e-10, atol=1e-10
        )

    def test_get_yaw(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )

        yaw = attitude.get_yaw(self.time_axis.origin + 3.5)
        yaw_reference = 1.451920577074731e-05
        np.testing.assert_allclose(yaw, yaw_reference, rtol=1e-10, atol=1e-16)

    def test_get_pitch(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )

        pitch = attitude.get_pitch(self.time_axis.origin + 3.5)
        pitch_reference = 6.718901618413190e-06
        np.testing.assert_allclose(pitch, pitch_reference, rtol=1e-10, atol=1e-16)

    def test_get_roll(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )

        roll = attitude.get_roll(self.time_axis.origin + 3.5)
        roll_reference = 25.925585457073900
        np.testing.assert_allclose(roll, roll_reference, rtol=1e-10, atol=1e-16)

    def test_sat2earthLOS(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )
        time_point = attitude.t0 + 3
        time_inputs: List[Union[PreciseDateTime, np.ndarray]] = [
            time_point,
            np.tile(self.time_axis.origin + 3, (10,)),
        ]

        az_angles_in = np.deg2rad(np.linspace(-5, 5, 10))
        el_angles_in = np.deg2rad(np.linspace(-3, 2, 10))

        azimuth_angles_inputs = [az_angles_in, az_angles_in[0]]
        elevation_angles_inputs = [el_angles_in, el_angles_in[4]]
        altitude_inputs = [0, 1000, -1000]

        for time, az, el, height in itertools.product(
            time_inputs, azimuth_angles_inputs, elevation_angles_inputs, altitude_inputs
        ):
            points = attitude.sat2earthLOS(time, az, el, altitude_over_wgs84=height)
            expected_shape = (3,)
            if not (
                isinstance(time, PreciseDateTime)
                and isinstance(az, float)
                and isinstance(el, float)
            ):
                expected_shape = (
                    max(np.size(time), np.size(az), np.size(el)),  # type: ignore
                ) + expected_shape
            self.assertEqual(points.shape, expected_shape)

            arf = attitude.get_arf(time)
            sensor_positions = self.orbit.get_position(time).T
            los = points - sensor_positions
            azimuth_out, elevation_out = _compute_antenna_angles_a_posteriori(arf, los)

            self.assertLess(np.max(np.abs(azimuth_out - az)), 1e-10)
            self.assertLess(np.max(np.abs(elevation_out - el)), 1e-10)

    def test_sat2earthLOS_invalid_inputs(self):
        attitude = GeneralSarAttitude(
            self.orbit, self.time_axis, self.ypr_matrix, "YPR", "ZERODOPPLER"
        )
        time_point = self.time_axis.origin + 3
        time_inputs = [
            time_point,
            np.tile(self.time_axis.origin + 3, (10,)),
        ]

        az_angles_in = np.deg2rad(np.linspace(-5, 5, 10))
        el_angles_in = np.deg2rad(np.linspace(-3, 2, 10))

        azimuth_inputs = [az_angles_in, az_angles_in[0:2]]
        elevation_inputs = [el_angles_in[0:2], el_angles_in[4]]

        for time, az, el in zip(time_inputs, azimuth_inputs, elevation_inputs):
            with self.assertRaises(ValueError):
                attitude.sat2earthLOS(time, az, el)

    def test_compute_pointing_directions(self):
        arf_in = np.array(
            [
                [-0.604580222175426, -0.564852727629684, -0.561626344684451],
                [-0.304829433093801, 0.815474732550541, -0.492016236816769],
                [0.735908806628936, -0.126263045507894, -0.665203631728697],
            ]
        )

        boresight_dir = compute_pointing_directions(arf_in, 0, 0)
        np.testing.assert_allclose(boresight_dir, arf_in[:, 2], rtol=1e-10, atol=1e-10)

        num_elements = 10
        arf_inputs = [arf_in, np.tile(arf_in, (num_elements, 1, 1))]

        az_angles_in = np.deg2rad(np.linspace(-5, 5, 10))
        el_angles_in = np.deg2rad(np.linspace(-3, 2, 10))

        azimuth_angles_inputs = [az_angles_in, az_angles_in[0]]
        elevation_angles_inputs = [el_angles_in, el_angles_in[4]]

        for arf, azimuth_angles, elevation_angles in itertools.product(
            arf_inputs, azimuth_angles_inputs, elevation_angles_inputs
        ):
            directions = compute_pointing_directions(
                arf, azimuth_angles, elevation_angles
            )

            expected_shape = (3,)
            if (
                arf.ndim == 3
                or np.size(azimuth_angles) > 1
                or np.size(elevation_angles) > 1
            ):
                expected_shape = (num_elements,) + expected_shape

            self.assertEqual(directions.shape, expected_shape)

            np.testing.assert_allclose(
                np.linalg.norm(directions, axis=-1),
                np.ones_like(azimuth_angles),
                rtol=1e-10,
                atol=1e-10,
            )

            azimuth_out, elevation_out = _compute_antenna_angles_a_posteriori(
                arf_in, directions
            )

            self.assertLess(np.max(np.abs(azimuth_out - azimuth_angles)), 1e-10)
            self.assertLess(np.max(np.abs(elevation_out - elevation_angles)), 1e-10)

    def test_compute_pointing_directions_invalid_inputs(self):
        arf_inputs = [
            np.ones((10, 3, 3)),
            np.ones((10, 3, 3)),
            np.ones((10, 3, 3)),
            np.ones((10, 3, 3)),
            np.ones((3, 3)),
            np.ones((3, 2)),
            np.ones((10, 3, 2)),
        ]
        azimuth_angles_inputs = [
            np.arange(5),
            np.arange(5),
            1,
            np.arange(10),
            np.arange(5),
            1,
            1,
        ]
        elevation_angles_inputs = [
            np.arange(5),
            1,
            np.arange(5),
            np.arange(3),
            np.arange(3),
            1,
        ]

        for arf, azimuth_angles, elevation_angles in zip(
            arf_inputs, azimuth_angles_inputs, elevation_angles_inputs
        ):
            with self.assertRaises(ValueError):
                compute_pointing_directions(arf, azimuth_angles, elevation_angles)


if __name__ == "__main__":
    unittest.main()
