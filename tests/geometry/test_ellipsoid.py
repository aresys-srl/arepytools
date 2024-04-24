# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import FrozenInstanceError
from unittest import TestCase

import numpy as np

from arepytools.geometry.ellipsoid import (
    WGS84,
    Ellipsoid,
    compute_line_ellipsoid_intersections,
)


class EllipsoidTestCase(TestCase):
    def test_init(self):
        self.assertRaises(TypeError, Ellipsoid)
        self.assertRaises(TypeError, Ellipsoid, 1)
        self.assertRaises(TypeError, Ellipsoid, (1, 2, 3))

    def test_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            e = Ellipsoid(1, 1)
            e.eccentricity = 10

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Ellipsoid(-1, 2)
        with self.assertRaises(ValueError):
            Ellipsoid(1, -2)
        with self.assertRaises(ValueError):
            Ellipsoid(-1, -2)

    def test_inflate(self):
        ellipsoid = Ellipsoid(1, 2)

        inflated_ellipsoid = ellipsoid.inflate(5)
        self.assertEqual(inflated_ellipsoid.semi_major_axis, 7)
        self.assertEqual(inflated_ellipsoid.semi_minor_axis, 6)

        inflated_ellipsoid = ellipsoid.inflate(0)
        self.assertEqual(inflated_ellipsoid.semi_major_axis, ellipsoid.semi_major_axis)
        self.assertEqual(inflated_ellipsoid.semi_minor_axis, ellipsoid.semi_minor_axis)

        inflated_ellipsoid = ellipsoid.inflate(-0.5)
        self.assertEqual(inflated_ellipsoid.semi_major_axis, 1.5)
        self.assertEqual(inflated_ellipsoid.semi_minor_axis, 0.5)

        with self.assertRaises(ValueError):
            ellipsoid.inflate(-1)


class LineEllipsoidIntersection(TestCase):
    def test_compute_line_ellipsoid_intersection_no_intersections(self):
        line_origin = [-5, 0, 2]
        line_direction = [0, 0, 100]
        ellipsoid = Ellipsoid(1, 2)

        intersections = compute_line_ellipsoid_intersections(
            line_direction, line_origin, ellipsoid
        )

        self.assertEqual(len(intersections), 0)

    def test_compute_line_ellipsoid_intersection_one_intersection(self):
        line_origin = [-5, 0, 1]
        line_direction = [100, 0, 0]
        ellipsoid = Ellipsoid(1, 2)

        intersections = compute_line_ellipsoid_intersections(
            line_direction, line_origin, ellipsoid
        )

        self.assertEqual(len(intersections), 1)
        np.testing.assert_allclose(
            intersections[0], np.array([0, 0, 1]), rtol=1e-10, atol=1e-10
        )

    def test_compute_line_ellipsoid_intersection_two_intersection(self):
        line_origin = [-5, 0, 0]
        line_direction = [100, 0, 0]
        ellipsoid = Ellipsoid(1, 2)

        intersections = compute_line_ellipsoid_intersections(
            line_direction, line_origin, ellipsoid
        )

        self.assertEqual(len(intersections), 2)

        self.assertLess(
            np.linalg.norm(intersections[0] - line_origin),
            np.linalg.norm(intersections[1] - line_origin),
        )

        np.testing.assert_allclose(
            intersections[0], np.array([-2, 0, 0]), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            intersections[1], np.array([2, 0, 0]), rtol=1e-10, atol=1e-10
        )

        line_origin = [0, 2, 0]
        line_direction = [0, 1, 0]
        ellipsoid = Ellipsoid(1, 2)

        intersections = compute_line_ellipsoid_intersections(
            line_direction, line_origin, ellipsoid
        )

        self.assertEqual(len(intersections), 2)

        self.assertLess(
            np.linalg.norm(intersections[0] - line_origin),
            np.linalg.norm(intersections[1] - line_origin),
        )

        np.testing.assert_allclose(
            intersections[0], np.array([0, 2, 0]), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            intersections[1], np.array([0, -2, 0]), rtol=1e-10, atol=1e-10
        )

        line_origin = [[0, 0, 10]]
        line_direction = [[0, 0, 5]]
        ellipsoid = Ellipsoid(1, 2)

        intersections = compute_line_ellipsoid_intersections(
            line_direction, line_origin, ellipsoid
        )
        intersections = intersections[0]

        self.assertEqual(len(intersections), 2)

        self.assertLess(
            np.linalg.norm(intersections[0] - line_origin),
            np.linalg.norm(intersections[1] - line_origin),
        )

        np.testing.assert_allclose(
            intersections[0], np.array([0, 0, 1]), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            intersections[1], np.array([0, 0, -1]), rtol=1e-10, atol=1e-10
        )

    def test_compute_line_ellipsoid_interseciton_vectorized(self):
        line_origins = np.array([[-5, 0, 2], [-5, 0, 1], [-5, 0, 0]])
        line_directions = np.array([[0, 0, 100], [100, 0, 0], [100, 0, 0]])
        ellipsoid = Ellipsoid(1, 2)

        intersections = compute_line_ellipsoid_intersections(
            line_directions, line_origins, ellipsoid
        )
        self.assertEqual(len(intersections), 3)

        self.assertEqual(len(intersections[0]), 0)
        self.assertEqual(len(intersections[1]), 1)
        np.testing.assert_allclose(
            intersections[1][0], np.array([0, 0, 1]), rtol=1e-10, atol=1e-10
        )

        self.assertLess(
            np.linalg.norm(intersections[2][0] - line_origins[2]),
            np.linalg.norm(intersections[2][1] - line_origins[2]),
        )

        np.testing.assert_allclose(
            intersections[2][0], np.array([-2, 0, 0]), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            intersections[2][1], np.array([2, 0, 0]), rtol=1e-10, atol=1e-10
        )

    def test_compute_line_ellipsoid_intersection_invalid_input(self):
        line_origin = [-5, 0]
        line_direction = [0, 0, 100]
        ellipsoid = Ellipsoid(1, 2)

        with self.assertRaises(ValueError):
            compute_line_ellipsoid_intersections(line_direction, line_origin, ellipsoid)

        line_direction = [0, 0, 100, 0]
        with self.assertRaises(ValueError):
            compute_line_ellipsoid_intersections(line_direction, line_origin, ellipsoid)


class TestWGS84(TestCase):
    def test_amax(self):
        self.assertEqual(WGS84.semi_major_axis, 6378137.0)

    def test_amin(self):
        self.assertEqual(WGS84.semi_minor_axis, 6356752.314245)

    def test_ratio(self):
        self.assertEqual(WGS84.semi_axes_ratio_min_max, 0.9966471893352243)

    def test_eccentricity(self):
        self.assertEqual(WGS84.eccentricity, 0.08181919084296488)

    def test_eccentricity_squared(self):
        self.assertEqual(WGS84.eccentricity_square, 0.006694379990197508)

    def test_ep2(self):
        self.assertEqual(WGS84.ep2, 0.006739496742333317)


if __name__ == "__main__":
    unittest.main()
