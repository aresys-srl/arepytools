# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

from arepytools.geometry._geodetic import (
    _compute_geodetic_jacobian,
    _compute_geodetic_rhs,
    compute_geodetic_point,
)


class GeodeticTestCase(unittest.TestCase):
    sensor_position = np.asarray(
        [26512.279931507, 1064819.379506800, 7083173.555337110]
    )
    sensor_velocity = np.asarray(
        [7529.609430015988, -342.978175622686, -23.376907795264]
    )

    point = np.asarray(
        [2.354782797513972e04, 9.457573485946435e05, 6.286436197788252e06]
    )

    def test_compute_geodetic_jacobian(self):
        jac = _compute_geodetic_jacobian(self.point, self.sensor_position)
        reference = np.asarray(
            [
                [0.000000001157692, 0.000000046496690, 0.000000311145789],
                [1.125892435250031, 0.000070706047773, -0.003733256145132],
                [0.000070706047773, 1.128730459597184, -0.149939707270331],
            ],
            dtype=float,
        )

        np.testing.assert_allclose(jac, reference, rtol=1e-10, atol=1e-10)

    def test_compute_geodetic_rhs(self):
        rhs = _compute_geodetic_rhs(self.point, self.sensor_position)
        reference = np.asarray(
            [0, 9.972679758107461, 4.005352500563895e02],
            dtype=float,
        )

        np.testing.assert_allclose(rhs, reference, rtol=1e-10, atol=1e-10)

    def test_compute_geodetic_point(self):
        input = self.sensor_position.copy()
        input[2] *= -1

        point = compute_geodetic_point(input)
        reference = np.asarray(
            [2.353916784173231e04, 9.454095294744077e05, -6.286488197273256e06]
        )

        np.testing.assert_allclose(point, reference, rtol=1e-10, atol=1e-10)

    def test_compute_geodetic_point_vectorized(self):
        input = self.sensor_position.copy()
        input[2] *= -1
        reference = np.asarray(
            [2.353916784173231e04, 9.454095294744077e05, -6.286488197273256e06]
        )

        point = compute_geodetic_point(input.reshape((1, 3)))

        np.testing.assert_allclose(
            point, reference.reshape((1, 3)), rtol=1e-10, atol=1e-10
        )

        point = compute_geodetic_point(np.tile(input, (2, 1)))

        np.testing.assert_allclose(
            point, np.tile(reference, (2, 1)), rtol=1e-10, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
