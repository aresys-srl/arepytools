# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Generic polynomial module
-------------------------
"""

from arepytools.io.metadata import _Poly2D, _Poly2DVector


class GenericPoly:
    """Generic polynomial class"""

    def __init__(self, reference_values, coefficients, powers):
        """Initialize the GenericPoly class

        .. math:: y(t) = \\sum_{i=0}^{d-1} c_i \\prod_{k=0}^{n-1}(t_k-t_{ref,k})^{p_{d,k}}

        where :math:`c_i` are the coefficients, :math:`t_{ref,k}` are the reference values and
        :math:`p_{d,k}` are the power exponents.

        Parameters
        ----------
        reference_values : Sequence[Any]
            n referenve value, one for each polynomial dimension
        coefficients : Sequence[float]
            d polynomials coefficients one for each term of the polynomial
        powers : Sequence[Sequence[int]]
            d items, one for each term of the polynomial; each item has n values corresponding to the exponents
            associated to each dimension in the current term
        """

        self.reference_values = reference_values
        self.poly = [
            (coefficient, powers[index_coefficients])
            for index_coefficients, coefficient in enumerate(coefficients)
        ]

    def __repr__(self):
        coefficients = [coeff for coeff, _ in self.poly]
        powers = [powers for _, powers in self.poly]
        reference_values = list(self.reference_values)
        representation = f"coefficients: {coefficients}\n"
        representation += f"powers: {powers}\n"
        representation += f"reference values: {reference_values}\n"
        return representation

    def evaluate(self, values):
        """Evaluate the GenericPoly for the values provided

        Parameters
        ----------
        values : Sequence[Any]
            a point (with n component) where to evaluate the polinomial

        Returns
        -------
        float
            interpolation result
        """
        values = tuple(values)
        result = 0
        for coefficient, powers in self.poly:
            current_result = 1
            for index_dimensions, ref_val in enumerate(self.reference_values):
                current_result *= (values[index_dimensions] - ref_val) ** powers[
                    index_dimensions
                ]
            result += coefficient * current_result

        return result


class SortedPolyList:
    """SortedPolyList class, a composite polynomial"""

    def __init__(self, reference_index=0, list_generic_poly=None):
        """Initialize sorted poly list

        Parameters
        ----------
        reference_index : int, optional
            sorting index, by default 0
        list_generic_poly : Sequence[GenericPoly] , optional
            list of generic poly, by default None
        """
        self.reference_index = reference_index
        if list_generic_poly is not None:
            self._sorted_poly_list = list_generic_poly
            self._sort_poly_list()
        else:
            self._sorted_poly_list = []

    def __repr__(self):
        representation = "Sorted Poly List:\n"
        for index, poly in enumerate(self._sorted_poly_list):
            representation += f"Poly #{index}: \n{poly}\n"
        return representation

    def append(self, generic_poly):
        """Append a new poly to the SortedPolyList

        Parameters
        ----------
        generic_poly : GenericPoly
            a new generic poly
        """
        self._sorted_poly_list.append(generic_poly)
        self._sort_poly_list()

    def _sort_poly_list(self):
        self._sorted_poly_list.sort(
            key=lambda x: x.reference_values[self.reference_index]
        )

    def evaluate(self, values):
        """Evaluate the composite polynomial for the values provided

        Parameters
        ----------
        values : Sequence[Any]
            a point (with n component) where to evaluate the polinomial list

        Returns
        -------
        float
            interpolation result
        """
        previous_poly = self._sorted_poly_list[0]
        for poly in self._sorted_poly_list:
            if (
                poly.reference_values[self.reference_index]
                > values[self.reference_index]
            ):
                break
            previous_poly = poly
        return previous_poly.evaluate(values)


def _create_generic_poly(poly2d: _Poly2D):
    """Create a GenericPoly from the values provided in metadata classes of base type _Poly2D"""
    return GenericPoly(
        reference_values=[poly2d.t_ref_az, poly2d.t_ref_rg],
        coefficients=poly2d.coefficients,
        powers=list(zip(poly2d.get_powers_x(), poly2d.get_powers_y())),
    )


def create_sorted_poly_list(poly2d_vector: _Poly2DVector):
    """Create a SortedPolyList from the values provided in metadata classes of base type _Poly2DVector

    Parameters
    ----------
    poly2d_vector : _Poly2DVector
        metadata representing a list of polynomials

    Returns
    -------
    SortedPolyList
        the corresponding composite polynomial
    """
    return SortedPolyList(
        list_generic_poly=[_create_generic_poly(p) for p in poly2d_vector]
    )
