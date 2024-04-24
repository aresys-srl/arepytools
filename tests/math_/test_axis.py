# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

from unittest import TestCase

import numpy as np

from arepytools.math.axis import Axis, RegularAxis
from arepytools.timing.precisedatetime import PreciseDateTime


def check_get_array(test_class, reference_axis, axis, size):
    for val, ref in zip(reference_axis, axis.get_array()):
        test_class.assertEqual(ref, val)
    for val, ref in zip(reference_axis[5:], axis.get_array(5)):
        test_class.assertEqual(ref, val)
    for val, ref in zip(reference_axis[5:18], axis.get_array(5, 18)):
        test_class.assertEqual(ref, val)
    test_class.assertRaises(ValueError, axis.get_array, -1, 3)
    test_class.assertRaises(ValueError, axis.get_array, size - 3, size + 2)


def check_get_interval_id(test_class, axis, ax_step, ax_length):
    sign = 1 if axis.increasing else -1
    ax_start = axis.start
    ax_size = axis.size

    test_class.assertEqual(axis.get_interval_id(ax_start - sign * ax_start / 2), 0)
    test_class.assertEqual(
        axis.get_interval_id(ax_start + sign * ax_length + ax_step / 2), ax_size - 1
    )
    test_class.assertEqual(
        axis.get_interval_id(ax_start + sign * ax_length / 2), np.ceil(ax_size / 2 - 1)
    )

    values = np.array(
        [
            ax_start - sign * ax_start,
            ax_start,
            ax_start + 6.2 * ax_step,
            ax_start + sign * ax_length * 2,
            ax_start + ax_step * 5.999999,
        ]
    )
    expected_res = np.array([0, 0, 6, ax_size - 1, 5])
    res = axis.get_interval_id(values)
    for val, ref in zip(res, expected_res):
        test_class.assertEqual(ref, val)


class TestRegularAxis(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = 3.14
        self.step = 0.1
        self.size = 100
        self.array = np.array([self.start + self.step * k for k in range(self.size)])
        self.basic = RegularAxis((self.start, self.step, self.size), 0)
        self.length = (self.size - 1) * self.step

        self.step_negative = -self.step
        self.array_negative = np.array(
            [self.start + self.step_negative * k for k in range(self.size)]
        )
        self.size_negative = self.array.size
        self.length_negative = abs(self.array_negative[0] - self.array_negative[-1])
        self.regular_negative = RegularAxis(
            (self.start, self.step_negative, self.size_negative), 0
        )

    def test_start(self):
        self.assertEqual(self.start, self.basic.start)
        self.assertEqual(self.start, self.regular_negative.start)

    def test_step(self):
        self.assertEqual(self.step, self.basic.step)
        self.assertEqual(self.step_negative, self.regular_negative.step)

    def test_size(self):
        self.assertEqual(self.size, self.basic.size)
        self.assertEqual(self.size_negative, self.regular_negative.size)

    def test_increasing(self):
        self.assertTrue(self.basic.increasing)
        self.assertFalse(self.regular_negative.increasing)

    def test_decreasing(self):
        self.assertFalse(self.basic.decreasing)
        self.assertTrue(self.regular_negative.decreasing)

    def test_length(self):
        self.assertEqual(self.length, self.basic.length)
        self.assertEqual(self.length_negative, self.regular_negative.length)

    def test_get_array(self):
        check_get_array(self, self.array, self.basic, self.size)
        check_get_array(
            self, self.array_negative, self.regular_negative, self.size_negative
        )

    def test_get_interval_id(self):
        check_get_interval_id(self, self.basic, self.step, self.length)
        check_get_interval_id(
            self, self.regular_negative, self.step_negative, self.length_negative
        )

    def test_interpolate(self):
        zoom = 0.1
        for t, ref in zip(
            self.basic.interpolate(np.arange(0, self.size - 1, zoom)),
            [self.start + self.step * k * zoom for k in range(self.size)],
        ):
            self.assertEqual(t, ref)
        for t, ref in zip(
            self.regular_negative.interpolate(np.arange(0, self.size - 1, zoom)),
            [
                self.start + self.step_negative * k * zoom
                for k in range(self.size_negative)
            ],
        ):
            self.assertEqual(t, ref)


class TestGeneralAxis(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.step = 0.1
        self.start = 10
        self.stop = 40

        self.array = np.arange(self.start, self.stop, self.step)
        self.size = self.array.size
        self.array[1::2] += self.step * 1.0e-4
        self.length = self.array[-1] - self.array[0]
        self.general = Axis(self.array)

        self.step_negative = -self.step
        self.array_negative = (
            -np.arange(self.start, self.stop, -self.step_negative) + 2 * self.start
        )
        self.size_negative = self.array.size
        self.array_negative[1::2] += self.step * 1.0e-4
        self.length_negative = abs(self.array_negative[0] - self.array_negative[-1])
        self.general_negative = Axis(self.array_negative, 0)

    def test_start(self):
        self.assertEqual(self.start, self.general.start)
        self.assertEqual(self.start, self.general_negative.start)

    def test_mean_step(self):
        self.assertAlmostEqual(
            self.step, self.general.mean_step, delta=self.step * 0.0001
        )
        self.assertAlmostEqual(
            self.step_negative,
            self.general_negative.mean_step,
            delta=self.step * 0.0001,
        )

    def test_size(self):
        self.assertEqual(self.size, self.general.size)
        self.assertEqual(self.size_negative, self.general_negative.size)

    def test_increasing(self):
        self.assertTrue(self.general.increasing)
        self.assertFalse(self.general_negative.increasing)

    def test_decreasing(self):
        self.assertFalse(self.general.decreasing)
        self.assertTrue(self.general_negative.decreasing)

    def test_length(self):
        self.assertEqual(self.length, self.general.length)
        self.assertEqual(self.length_negative, self.general_negative.length)

    def test_get_array(self):
        tolerance = 0
        np.testing.assert_allclose(
            self.general.get_array(), self.array, atol=tolerance, rtol=0
        )
        np.testing.assert_allclose(
            self.general.get_array(5), self.array[5:], atol=tolerance, rtol=0
        )
        np.testing.assert_allclose(
            self.general.get_array(5, 18), self.array[5:18], atol=tolerance, rtol=0
        )
        self.assertRaises(ValueError, self.general.get_array, -1, 3)
        self.assertRaises(
            ValueError, self.general.get_array, self.size - 3, self.size + 2
        )

        np.testing.assert_allclose(
            self.general_negative.get_array(),
            self.array_negative,
            atol=tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self.general_negative.get_array(5),
            self.array_negative[5:],
            atol=tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self.general_negative.get_array(5, 18),
            self.array_negative[5:18],
            atol=tolerance,
            rtol=0,
        )
        self.assertRaises(ValueError, self.general_negative.get_array, -1, 3)
        self.assertRaises(
            ValueError,
            self.general_negative.get_array,
            self.size_negative - 3,
            self.size_negative + 2,
        )

    def test_get_interval_id(self):
        check_get_interval_id(self, self.general, self.step, self.length)
        check_get_interval_id(
            self, self.general_negative, self.step_negative, self.length_negative
        )

    def test_interpolate(self):
        for t, ref in zip(
            self.general.interpolate(np.arange(0, self.size - 1)), self.array
        ):
            self.assertEqual(t, ref)
        for t, ref in zip(
            self.general_negative.interpolate(np.arange(0, self.size - 1)),
            self.array_negative,
        ):
            self.assertEqual(t, ref)


class TestAzimuthRegularAxis(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.az_start = PreciseDateTime.now()
        self.step = 0.1
        self.size = 100
        self.array = np.arange(0, self.step * self.size, self.step)
        self.az_regular = RegularAxis((0, self.step, self.size), self.az_start)

    def test_start(self):
        self.assertEqual(0, self.az_start - self.az_regular.start)

    def test_get_relative_array(self):
        array = self.az_regular.get_relative_array()
        np.testing.assert_allclose(
            array, self.array, atol=PreciseDateTime.get_precision(), rtol=0
        )

        array = self.az_regular.get_relative_array(5, 10)
        np.testing.assert_allclose(
            array, self.array[5:10], atol=PreciseDateTime.get_precision(), rtol=0
        )

        array = self.az_regular.get_relative_array(5)
        np.testing.assert_allclose(
            array, self.array[5:], atol=PreciseDateTime.get_precision(), rtol=0
        )

        array = self.az_regular.get_relative_array(0, self.array.size)
        np.testing.assert_allclose(
            array, self.array, atol=PreciseDateTime.get_precision(), rtol=0
        )


class TestAzimuthGeneralAxis(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.step = 0.1
        self.start = 10
        self.stop = 40

        self.array = np.arange(self.start, self.stop, self.step)
        self.size = self.array.size
        self.array[1::2] += self.step * 1.0e-4
        self.length = self.array[-1] - self.array[0]
        self.azimuth_start = PreciseDateTime.from_utc_string(
            "09-NOV-2018 12:40:01.296499967575"
        )
        self.az_general = Axis(self.array, self.azimuth_start)

    def test_start(self):
        self.assertEqual(self.az_general.start, self.azimuth_start + self.array[0])

    def test_get_relative_array(self):
        tolerance = 0

        array = self.az_general.get_relative_array()
        np.testing.assert_allclose(array, self.array, atol=tolerance, rtol=0)

        array = self.az_general.get_relative_array(5, 10)
        np.testing.assert_allclose(array, self.array[5:10], atol=tolerance, rtol=0)

        array = self.az_general.get_relative_array(5)
        np.testing.assert_allclose(array, self.array[5:], atol=tolerance, rtol=0)

        array = self.az_general.get_relative_array(0, self.array.size)
        np.testing.assert_allclose(array, self.array, atol=tolerance, rtol=0)

    def test_get_array(self):
        array = self.az_general.get_array()
        tolerance = 0
        np.testing.assert_allclose(
            (array - self.azimuth_start).astype(float),
            self.array,
            atol=tolerance,
            rtol=0,
        )

        array = self.az_general.get_array(5, 10)
        np.testing.assert_allclose(
            (array - self.azimuth_start).astype(float),
            self.array[5:10],
            atol=tolerance,
            rtol=0,
        )

        array = self.az_general.get_array(5)
        np.testing.assert_allclose(
            (array - self.azimuth_start).astype(float),
            self.array[5:],
            atol=tolerance,
            rtol=0,
        )

        array = self.az_general.get_array(0, self.array.size)
        np.testing.assert_allclose(
            (array - self.azimuth_start).astype(float),
            self.array,
            atol=tolerance,
            rtol=0,
        )

    def test_interpolate(self):
        interpolated = (
            self.az_general.interpolate(np.arange(self.size)) - self.az_general.start
        ).astype(float)
        np.testing.assert_allclose(
            interpolated,
            (self.az_general.get_array() - self.az_general.start).astype(float),
            atol=PreciseDateTime.get_precision(),
            rtol=0,
        )
