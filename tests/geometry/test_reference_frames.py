# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

from arepytools.geometry.reference_frames import (
    ReferenceFrame,
    compute_geocentric_reference_frame,
    compute_geodetic_reference_frame,
    compute_inertial_velocity,
    compute_sensor_local_axis,
    compute_zerodoppler_reference_frame,
)


class SensorAxisTestCase(unittest.TestCase):
    sensor_position = np.asarray(
        [26512.279931507, 1064819.379506800, -7083173.555337110]
    )
    sensor_velocity = np.asarray(
        [7529.609430015988, -342.978175622686, -23.376907795264]
    )

    zerodoppler_frame_reference = np.asarray(
        [
            [0.998959378858231, 0.045461584707689, -0.003661107352025],
            [-0.045503192226166, 0.987972305556891, -0.147784244592682],
            [-0.003101433282541, 0.147797049254939, 0.989012847916106],
        ],
        dtype=float,
    )

    geocentric_frame_reference = np.asarray(
        [
            [0.998949483740530, 0.045675252972446, -0.003701378179995],
            [-0.045717707103110, 0.987831098247604, -0.148659384473924],
            [-0.003133718520000, 0.148672433896922, 0.988881533454540],
        ],
        dtype=float,
    )

    geodetic_frame_reference = np.asarray(
        [
            [0.998949449735728, 0.045676868615385, -0.003690602413993],
            [-0.045719074555789, 0.987896068814253, -0.148226594925155],
            [-0.003124595085362, 0.148239606363606, 0.988946538499789],
        ],
        dtype=float,
    )

    def test_compute_zerodoppler_reference_frame(self):
        frame = compute_zerodoppler_reference_frame(
            self.sensor_position, self.sensor_velocity
        )
        np.testing.assert_allclose(
            frame, self.zerodoppler_frame_reference, rtol=1e-10, atol=1e-10
        )

    def test_compute_zerodoppler_reference_frame_vectorized(self):
        frame = compute_zerodoppler_reference_frame(
            self.sensor_position.reshape((1, 3)),
            self.sensor_velocity.reshape((1, 3)),
        )

        np.testing.assert_allclose(
            frame,
            self.zerodoppler_frame_reference.reshape((1, 3, 3)),
            rtol=1e-10,
            atol=1e-10,
        )

        frame = compute_zerodoppler_reference_frame(
            np.tile(self.sensor_position, (2, 1)), np.tile(self.sensor_velocity, (2, 1))
        )

        np.testing.assert_allclose(
            frame,
            np.tile(self.zerodoppler_frame_reference, (2, 1, 1)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_compute_zerodoppler_reference_frame_invalid_input(self):
        with self.assertRaises(ValueError):
            compute_zerodoppler_reference_frame(np.ones((3, 10)), np.ones((3, 10)))

    def test_compute_geocentric_reference_frame(self):
        frame = compute_geocentric_reference_frame(
            self.sensor_position, self.sensor_velocity
        )
        np.testing.assert_allclose(
            frame, self.geocentric_frame_reference, rtol=1e-10, atol=1e-10
        )

    def test_compute_geocentric_reference_frame_vectorized(self):
        frame = compute_geocentric_reference_frame(
            self.sensor_position.reshape((1, 3)),
            self.sensor_velocity.reshape((1, 3)),
        )

        np.testing.assert_allclose(
            frame,
            self.geocentric_frame_reference.reshape((1, 3, 3)),
            rtol=1e-10,
            atol=1e-10,
        )

        frame = compute_geocentric_reference_frame(
            np.tile(self.sensor_position, (2, 1)), np.tile(self.sensor_velocity, (2, 1))
        )

        np.testing.assert_allclose(
            frame,
            np.tile(self.geocentric_frame_reference, (2, 1, 1)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_compute_geocentric_reference_frame_invalid_input(self):
        with self.assertRaises(ValueError):
            compute_geocentric_reference_frame(np.ones((10, 3)), np.ones((7, 3)))

        with self.assertRaises(ValueError):
            compute_geocentric_reference_frame(np.ones((3, 10)), np.ones((3, 10)))

    def test_compute_geodetic_reference_frame(self):
        frame = compute_geodetic_reference_frame(
            self.sensor_position, self.sensor_velocity
        )
        np.testing.assert_allclose(
            frame, self.geodetic_frame_reference, rtol=1e-10, atol=1e-10
        )

    def test_compute_geodetic_reference_frame_vectorized(self):
        frame = compute_geodetic_reference_frame(
            self.sensor_position.reshape((1, 3)), self.sensor_velocity.reshape((1, 3))
        )
        np.testing.assert_allclose(
            frame,
            self.geodetic_frame_reference.reshape((1, 3, 3)),
            rtol=1e-10,
            atol=1e-10,
        )

        frame = compute_geodetic_reference_frame(
            np.tile(self.sensor_position, (2, 1)), np.tile(self.sensor_velocity, (2, 1))
        )
        np.testing.assert_allclose(
            frame,
            np.tile(self.geodetic_frame_reference, (2, 1, 1)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_compute_geodetic_reference_frame_invalid_input(self):
        with self.assertRaises(ValueError):
            compute_geodetic_reference_frame(np.ones((10, 3)), np.ones((7, 3)))

        with self.assertRaises(ValueError):
            compute_geodetic_reference_frame(np.ones((3, 10)), np.ones((3, 10)))

    def test_compute_sensor_local_axis(self):
        frame = compute_sensor_local_axis(
            self.sensor_position, self.sensor_velocity, "ZERODOPPLER"
        )
        np.testing.assert_allclose(
            frame, self.zerodoppler_frame_reference, rtol=1e-10, atol=1e-10
        )

        frame = compute_sensor_local_axis(
            self.sensor_position, self.sensor_velocity, ReferenceFrame.zero_doppler
        )
        np.testing.assert_allclose(
            frame, self.zerodoppler_frame_reference, rtol=1e-10, atol=1e-10
        )

        frame = compute_sensor_local_axis(
            self.sensor_position, self.sensor_velocity, "GEOCENTRIC"
        )
        np.testing.assert_allclose(
            frame, self.geocentric_frame_reference, rtol=1e-10, atol=1e-10
        )

        frame = compute_sensor_local_axis(
            self.sensor_position, self.sensor_velocity, ReferenceFrame.geocentric
        )
        np.testing.assert_allclose(
            frame, self.geocentric_frame_reference, rtol=1e-10, atol=1e-10
        )

        frame = compute_sensor_local_axis(
            self.sensor_position, self.sensor_velocity, "GEODETIC"
        )
        np.testing.assert_allclose(
            frame, self.geodetic_frame_reference, rtol=1e-10, atol=1e-10
        )

        frame = compute_sensor_local_axis(
            self.sensor_position, self.sensor_velocity, ReferenceFrame.geodetic
        )
        np.testing.assert_allclose(
            frame, self.geodetic_frame_reference, rtol=1e-10, atol=1e-10
        )

    def test_compute_sensor_local_axis_invalid_reference_frame(self):
        with self.assertRaises(ValueError):
            compute_sensor_local_axis(self.sensor_position, self.sensor_velocity, None)

        with self.assertRaises(ValueError):
            compute_sensor_local_axis(
                self.sensor_position, self.sensor_velocity, "unknown name"
            )

        with self.assertRaises(ValueError):
            compute_sensor_local_axis(
                self.sensor_position, self.sensor_velocity, "geocentric"
            )


class IntertialFramesTestCase(unittest.TestCase):
    sensor_position = np.asarray(
        [26512.279931507, 1064819.379506800, 7083173.555337110]
    )
    sensor_velocity = np.asarray(
        [7529.609430015988, -342.978175622686, -23.376907795264]
    )
    intertial_velocity_reference = np.asarray(
        [7.451961567220857e03, -3.410448694544030e02, -23.376907795264000]
    )

    def test_compute_intertial_velocity(self):
        intertial_velocity = compute_inertial_velocity(
            self.sensor_position, self.sensor_velocity
        )
        np.testing.assert_allclose(
            intertial_velocity,
            self.intertial_velocity_reference,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_compute_intertial_velocity_Vectorized(self):
        intertial_velocity = compute_inertial_velocity(
            self.sensor_position.reshape((1, 3)), self.sensor_velocity.reshape((1, 3))
        )
        np.testing.assert_allclose(
            intertial_velocity,
            self.intertial_velocity_reference.reshape((1, 3)),
            rtol=1e-10,
            atol=1e-10,
        )

        intertial_velocity = compute_inertial_velocity(
            np.tile(self.sensor_position, (2, 1)), np.tile(self.sensor_velocity, (2, 1))
        )
        np.testing.assert_allclose(
            intertial_velocity,
            np.tile(self.intertial_velocity_reference, (2, 1)),
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
