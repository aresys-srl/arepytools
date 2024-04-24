# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import dataclass, field
from typing import List, Tuple
from unittest import TestCase

from arepytools.io.metadata import _Poly2D, _Poly2DVector
from arepytools.math.genericpoly import (
    GenericPoly,
    SortedPolyList,
    _create_generic_poly,
    create_sorted_poly_list,
)
from arepytools.timing.precisedatetime import PreciseDateTime


@dataclass(frozen=True)
class PolyData:
    ref_az: PreciseDateTime
    ref_rg: float
    coefficients: List[float] = field(default_factory=list)
    powers: List[Tuple[int, int]] = field(default_factory=list)


poly_data_sorted_first = PolyData(
    ref_az=PreciseDateTime(),
    ref_rg=8,
    coefficients=[
        5.5,
        6.0,
        0.00000000e00,
        0.00000000e00,
        1.0,
        1.0,
    ],
    powers=[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 5)],
)

poly_data_sorted_second = PolyData(
    ref_az=PreciseDateTime.from_utc_string("09-JUL-2006 21:00:0.0"),
    ref_rg=2,
    coefficients=[
        2.5,
        2.5,
        0.0,
        -1.0,
    ],
    powers=[(0, 2), (1, 0), (1, 0), (0, 2)],
)


class TestGenericPoly(TestCase):
    def test_init(self):
        generic_poly = GenericPoly(
            reference_values=[
                poly_data_sorted_first.ref_az,
                poly_data_sorted_first.ref_rg,
            ],
            coefficients=poly_data_sorted_first.coefficients,
            powers=poly_data_sorted_first.powers,
        )

        self.assertEqual(
            generic_poly.reference_values[0], poly_data_sorted_first.ref_az
        )
        self.assertEqual(
            generic_poly.reference_values[1], poly_data_sorted_first.ref_rg
        )

        self.assertEqual(
            [coeff for coeff, _ in generic_poly.poly],
            poly_data_sorted_first.coefficients,
        )
        self.assertEqual(
            [powers for _, powers in generic_poly.poly],
            poly_data_sorted_first.powers,
        )

    def test_evaluate(self):
        generic_poly = GenericPoly(
            reference_values=[
                poly_data_sorted_second.ref_az,
                poly_data_sorted_second.ref_rg,
            ],
            coefficients=poly_data_sorted_second.coefficients,
            powers=poly_data_sorted_second.powers,
        )

        evaluation_values = [
            PreciseDateTime.from_utc_string("09-JUL-2006 21:00:3.0"),
            4,
        ]
        expected_result = 13.5

        result = generic_poly.evaluate(evaluation_values)

        self.assertEqual(result, expected_result)

    def test_create_generic_poly(self):
        poly2D = _create_generic_poly(
            _Poly2D(
                poly_data_sorted_second.ref_az,
                poly_data_sorted_second.ref_rg,
                poly_data_sorted_second.coefficients,
            )
        )

        self.assertEqual(poly2D.reference_values[0], poly_data_sorted_second.ref_az)
        self.assertEqual(poly2D.reference_values[1], poly_data_sorted_second.ref_rg)

        self.assertEqual(
            [coeff for coeff, _ in poly2D.poly],
            poly_data_sorted_second.coefficients,
        )

    def test_repr(self):
        generic_poly = GenericPoly(
            reference_values=[
                poly_data_sorted_first.ref_az,
                poly_data_sorted_first.ref_rg,
            ],
            coefficients=poly_data_sorted_first.coefficients,
            powers=poly_data_sorted_first.powers,
        )

        expected_message = "coefficients: {}\n".format(
            [coeff for coeff in poly_data_sorted_first.coefficients]
        )
        expected_message += "powers: {}\n".format(
            [powers for powers in poly_data_sorted_first.powers]
        )
        expected_message += "reference values: {}\n".format(
            list([poly_data_sorted_first.ref_az, poly_data_sorted_first.ref_rg])
        )

        self.assertEqual(generic_poly.__repr__(), expected_message)


class TestSortedPolyList(TestCase):
    def test_create_sorted_poly_list(self):
        poly2D_second_after_sort = _Poly2D(
            i_ref_az=poly_data_sorted_second.ref_az,
            i_ref_rg=poly_data_sorted_second.ref_rg,
            i_coefficients=poly_data_sorted_second.coefficients,
        )

        poly2D_first_after_sort = _Poly2D(
            i_ref_az=poly_data_sorted_first.ref_az,
            i_ref_rg=poly_data_sorted_first.ref_rg,
            i_coefficients=poly_data_sorted_first.coefficients,
        )

        poly_list = _Poly2DVector([poly2D_second_after_sort, poly2D_first_after_sort])
        sorted_poly_list = create_sorted_poly_list(poly_list)

        self.assertEqual(
            [coeff for coeff, _ in sorted_poly_list._sorted_poly_list[0].poly],
            poly2D_first_after_sort._coefficients,
        )

        self.assertEqual(
            [coeff for coeff, _ in sorted_poly_list._sorted_poly_list[1].poly],
            poly2D_second_after_sort._coefficients,
        )

    def test_evaluate(self):
        first_values = [
            PreciseDateTime.from_utc_string("09-JUL-2006 21:00:3.0"),
            4,
        ]

        second_values = [
            PreciseDateTime.from_utc_string("09-JUL-2006 20:59:7.0"),
            4,
        ]

        poly_second_after_sort = GenericPoly(
            reference_values=[
                poly_data_sorted_second.ref_az,
                poly_data_sorted_second.ref_rg,
            ],
            coefficients=poly_data_sorted_second.coefficients,
            powers=poly_data_sorted_second.powers,
        )

        poly_first_after_sort = GenericPoly(
            reference_values=[
                poly_data_sorted_first.ref_az,
                poly_data_sorted_first.ref_rg,
            ],
            coefficients=poly_data_sorted_first.coefficients,
            powers=poly_data_sorted_first.powers,
        )
        generic_poly_list = [poly_second_after_sort, poly_first_after_sort]

        sorted_poly_list = SortedPolyList(list_generic_poly=generic_poly_list)

        result = sorted_poly_list.evaluate(first_values)
        first_expected_result = 13.5

        self.assertEqual(result, first_expected_result)

        result = sorted_poly_list.evaluate(second_values)
        second_expected_result = -1026.5

        self.assertEqual(result, second_expected_result)

    def test_append(self):
        poly_second_after_sort = GenericPoly(
            reference_values=[
                poly_data_sorted_second.ref_az,
                poly_data_sorted_second.ref_rg,
            ],
            coefficients=poly_data_sorted_second.coefficients,
            powers=poly_data_sorted_second.powers,
        )

        poly_first_after_sort = GenericPoly(
            reference_values=[
                poly_data_sorted_first.ref_az,
                poly_data_sorted_first.ref_rg,
            ],
            coefficients=poly_data_sorted_first.coefficients,
            powers=poly_data_sorted_first.powers,
        )

        sorted_poly_list = SortedPolyList(list_generic_poly=None)

        sorted_poly_list.append(poly_first_after_sort)
        sorted_poly_list.append(poly_second_after_sort)

        self.assertEqual(
            [coeff for coeff, _ in sorted_poly_list._sorted_poly_list[0].poly],
            [coeff for coeff, _ in poly_first_after_sort.poly],
        )

        self.assertEqual(
            [coeff for coeff, _ in sorted_poly_list._sorted_poly_list[1].poly],
            [coeff for coeff, _ in poly_second_after_sort.poly],
        )

    def test_repr(self):
        poly_second_after_sort = GenericPoly(
            reference_values=[
                poly_data_sorted_second.ref_az,
                poly_data_sorted_second.ref_rg,
            ],
            coefficients=poly_data_sorted_second.coefficients,
            powers=poly_data_sorted_second.powers,
        )

        poly_first_after_sort = GenericPoly(
            reference_values=[
                poly_data_sorted_first.ref_az,
                poly_data_sorted_first.ref_rg,
            ],
            coefficients=poly_data_sorted_first.coefficients,
            powers=poly_data_sorted_first.powers,
        )
        generic_poly_list = [poly_second_after_sort, poly_first_after_sort]

        sorted_poly_list = SortedPolyList(list_generic_poly=generic_poly_list)

        expected_message = "Sorted Poly List:\n"
        for index, poly in enumerate(generic_poly_list):
            expected_message += "Poly #{}: \n{}\n".format(index, poly)

        self.assertEqual(sorted_poly_list.__repr__(), expected_message)


if __name__ == "__main__":
    unittest.main()
