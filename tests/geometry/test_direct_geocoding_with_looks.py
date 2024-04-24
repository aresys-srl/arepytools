# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import itertools
import unittest

import numpy as np

from arepytools.geometry.direct_geocoding import (
    direct_geocoding_with_look_angles,
    direct_geocoding_with_looking_direction,
)


class DirectGeocodingTestCase(unittest.TestCase):
    sensor_position = np.array([5317606.94350283, 610603.985945038, 4577936.89859885])
    sensor_velocity = np.array(
        [-4.579225059127757, -2.318410347817750, 5.612873789290531]
    )
    looking_direction = np.array([5317606.0, 610603.0, 4577936.0])
    wrong_looking_direction = np.array(
        [-4.617335568769981, -2.318493268811102, 5.581386037561345]
    )

    def test_direct_geocoding_with_looking_direction(self):
        altitude_over_wgs84 = 1000.0

        reference_point = 1.0e06 * np.array(
            [4.809369353321205, 0.552244754877962, 4.140394297540082]
        )

        position_inputs = [self.sensor_position, self.sensor_position.reshape((1, 3))]
        dir_inputs = [self.looking_direction, self.looking_direction.reshape((1, 3))]

        for origin, direction in itertools.product(position_inputs, dir_inputs):
            point = direct_geocoding_with_looking_direction(
                origin, direction, altitude_over_wgs84=altitude_over_wgs84
            )

            self.assertEqual(
                point.shape, np.broadcast_shapes(origin.shape, direction.shape)
            )
            np.testing.assert_allclose(
                point, reference_point.reshape(point.shape), rtol=1e-10, atol=1e-10
            )

    def test_direct_geocoding_with_looking_direction_vectorized(self):
        directions = np.vstack(
            (
                np.tile(self.looking_direction, (5, 1)),
                self.wrong_looking_direction,
            )
        )

        altitude_over_wgs84 = 1000.0

        reference_point = 1.0e06 * np.array(
            [4.809369353321205, 0.552244754877962, 4.140394297540082]
        )

        points = direct_geocoding_with_looking_direction(
            self.sensor_position,
            directions,
            altitude_over_wgs84=altitude_over_wgs84,
        )

        np.testing.assert_allclose(
            points[0:5], np.tile(reference_point, (5, 1)), rtol=1e-10, atol=1e-10
        )
        self.assertTrue(all(np.isnan(points[5])))

        points = direct_geocoding_with_looking_direction(
            np.tile(self.sensor_position, (2, 1)),
            self.looking_direction,
            altitude_over_wgs84=altitude_over_wgs84,
        )

        np.testing.assert_allclose(
            points, np.tile(reference_point, (2, 1)), rtol=1e-10, atol=1e-10
        )

        points = direct_geocoding_with_looking_direction(
            np.tile(self.sensor_position, (2, 1)),
            np.tile(self.looking_direction, (2, 1)),
            altitude_over_wgs84=altitude_over_wgs84,
        )

        np.testing.assert_allclose(
            points, np.tile(reference_point, (2, 1)), rtol=1e-10, atol=1e-10
        )

    def test_direct_geocoding_with_looking_direction_empty_intersection(self):
        point = direct_geocoding_with_looking_direction(
            self.sensor_position, self.wrong_looking_direction
        )

        self.assertTrue(all(np.isnan(point)))

    def test_direct_geocoding_with_looking_direction_invalid_inptus(self):
        with self.assertRaises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(5), np.arange(3))
        with self.assertRaises(ValueError):
            direct_geocoding_with_looking_direction(
                np.arange(6).reshape(3, 2), np.arange(3)
            )
        with self.assertRaises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(3), np.arange(5))
        with self.assertRaises(ValueError):
            direct_geocoding_with_looking_direction(
                np.arange(3), np.arange(6).reshape(3, 2)
            )

    def test_direct_geocoding_with_look_angles(self):
        look_angles = np.deg2rad(np.arange(15, 50, 5))
        altitude = 0

        reference_points = 1.0e6 * np.array(
            [
                [4.759710115562946, 0.723739860905043, 4.169511582485821],
                [4.740767822131609, 0.785940591895703, 4.179749794811581],
                [4.719765088454115, 0.852956178108469, 4.190295797874668],
                [4.695901593234693, 0.926811645389169, 4.201333096521494],
                [4.668001681937613, 1.010339507423213, 4.213072517214957],
                [4.634228769201612, 1.107766560515324, 4.225761582103892],
                [4.591484972384566, 1.225899798385040, 4.239684548934286],
            ]
        )

        points = direct_geocoding_with_look_angles(
            self.sensor_position,
            self.sensor_velocity,
            "ZERODOPPLER",
            look_angles,
            altitude_over_wgs84=altitude,
        )

        np.testing.assert_allclose(points, reference_points, rtol=1e-10, atol=1e-10)

        points = direct_geocoding_with_look_angles(
            self.sensor_position.reshape(1, 3),
            self.sensor_velocity.reshape(1, 3),
            "ZERODOPPLER",
            look_angles,
            altitude_over_wgs84=altitude,
        )

        np.testing.assert_allclose(points, reference_points, rtol=1e-10, atol=1e-10)

        point = direct_geocoding_with_look_angles(
            self.sensor_position,
            self.sensor_velocity,
            "ZERODOPPLER",
            look_angles[0],
            altitude_over_wgs84=altitude,
        )

        np.testing.assert_allclose(point, reference_points[0], rtol=1e-10, atol=1e-10)

        points = direct_geocoding_with_look_angles(
            np.tile(self.sensor_position, (10, 1)),
            np.tile(self.sensor_velocity, (10, 1)),
            "ZERODOPPLER",
            np.tile(look_angles[0], (10,)),
            altitude_over_wgs84=altitude,
        )

        np.testing.assert_allclose(
            points, np.tile(reference_points[0], (10, 1)), rtol=1e-10, atol=1e-10
        )

    def test_direct_geocoding_with_look_angles_invalid_inptus(self):
        with self.assertRaises(ValueError):
            direct_geocoding_with_look_angles(
                np.arange(5, dtype=float),
                np.arange(3, dtype=float),
                "ZERODOPPLER",
                np.arange(5, dtype=float),
            )
        with self.assertRaises(ValueError):
            direct_geocoding_with_look_angles(
                np.arange(3, dtype=float),
                np.arange(4, dtype=float),
                "ZERODOPPLER",
                np.arange(5, dtype=float),
            )
        with self.assertRaises(ValueError):
            direct_geocoding_with_look_angles(
                np.arange(3, dtype=float),
                np.arange(3, dtype=float),
                "ZERODOPPLER",
                np.arange(6, dtype=float).reshape(2, 3),
            )


if __name__ == "__main__":
    unittest.main()
