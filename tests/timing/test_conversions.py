# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for timing.conversion module"""

import unittest
from datetime import datetime

import arepytools.timing.conversions as conv
from arepytools.timing.precisedatetime import PreciseDateTime


class GPSWeekConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_date = PreciseDateTime.from_numeric_datetime(2012, 6, 15, 17, 30)
        self.input_date_2 = datetime(1979, 6, 15, 17, 30)
        self.ref_results = (1692, 5)

    def test_date_to_gps_week_conversion(self) -> None:
        """Testing date_to_gps_week conversion function"""
        gps_week, day_of_week = conv.date_to_gps_week(self.input_date)

        self.assertIsInstance(gps_week, int)
        self.assertIsInstance(day_of_week, int)
        self.assertEqual(gps_week, self.ref_results[0])
        self.assertEqual(day_of_week, self.ref_results[1])

    def test_date_to_gps_week_conversion_with_error(self) -> None:
        """Testing date_to_gps_week conversion function, with error"""
        with self.assertRaises(ValueError):
            conv.date_to_gps_week(self.input_date_2)


if __name__ == "__main__":
    unittest.main()
