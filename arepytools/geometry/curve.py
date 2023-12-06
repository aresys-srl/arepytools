# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Curve classes implementation of curve protocols
-----------------------------------------------
"""

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from arepytools.geometry.curve_protocols import RealTwiceDifferentiableFunction
from arepytools.timing.precisedatetime import PreciseDateTime


class InterpolationOutOfBoundaries(ValueError):
    """Could not perform interpolation on input value because it's out of validity boundaries"""


class InvertedTimeBoundaries(RuntimeError):
    """Inverted time boundaries provided for interpolator domain definition"""


class PolynomialWrapper:
    """Polynomial Wrapper implementation"""

    def __init__(self, coeff: npt.ArrayLike) -> None:
        """Polynomial wrapper inputs.

        Polynomial in the form:

        .. math::

            a_{n}\\cdot x^{n} + a_{n-1}\\cdot x^{n-1} + ... + a_2\\cdot x^2 + a_1\\cdot x + a_0

        Parameters
        ----------
        coeff : npt.ArrayLike
            polynomial coefficients to build the polynomial interpolator,
            coefficients must be provided from the highers order to the lower
            the last coefficient is the constant term
        """
        self.poly_coefficients: np.ndarray = np.asarray(coeff)

    def evaluate(
        self, coords: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluating the input polynomial at the specified variable values.

        Polynomial in the form:

        .. math::

            a_{n}\\cdot x^{n} + a_{n-1}\\cdot x^{n-1} + ... + a_2\\cdot x^2 + a_1\\cdot x + a_0

        Parameters
        ----------
        coords : Union[float, npt.ArrayLike]
            values of the variable at which evaluate the polynomial

        Returns
        -------
        Union[float, npt.ArrayLike]
            polynomial computed at each input variable value
        """

        coords = np.asarray(coords).astype("float64")

        # evaluating variable powers for each term of the polynomial at the input coordinate
        variable_values = np.flip(
            [coords**c for c in range(self.poly_coefficients.size)], axis=0
        )

        return np.dot(self.poly_coefficients, variable_values)

    def evaluate_first_derivative(
        self, coords: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluating the input polynomial first derivative at the specified variable values.
        Polynomial derivative in the form:

        .. math::

            n\\cdot a_{n}\\cdot x^{n-1} + (n-1)\\cdot a_{n-1}\\cdot x^{n-2} + ... + 2\\cdot a_2\\cdot x + a_1

        Parameters
        ----------
        coords : Union[float, npt.ArrayLike]
            values of the variable at which evaluate the polynomial

        Returns
        -------
        Union[float, npt.ArrayLike]
            polynomial derivative computed at each input variable value
        """

        coords = np.asarray(coords).astype("float64")
        # evaluating the derivative polynomial n-coefficients: n*a^(n-1)
        poly_der_coeff = np.flip(range(1, self.poly_coefficients.size))
        coeff = self.poly_coefficients[:-1]
        # multiplying each polynomial variable term by its coefficient
        variable_values = np.flip(
            [coords**c for c in range(self.poly_coefficients.size - 1)], axis=0
        )

        # multiplying each polynomial variable term by its derivative coefficient and the polynomial coefficient
        return np.dot(poly_der_coeff * coeff, variable_values)

    def evaluate_second_derivative(
        self, coords: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluating the input polynomial second derivative at the specified variable values.

        Polynomial derivative in the form:

        .. math::

            n\\cdot (n-1)\\cdot a_{n}\\cdot x^{n-2} + (n-1)\\cdot (n-2)\\cdot a_{n-1}\\cdot x^{n-3} + ... + 2\\cdot a_2

        Parameters
        ----------
        coords : Union[float, npt.ArrayLike]
            values of the variable at which evaluate the polynomial

        Returns
        -------
        Union[float, npt.ArrayLike]
            polynomial derivative computed at each input variable value
        """

        coords = np.asarray(coords).astype("float64")
        # evaluating the derivative polynomial n-coefficients: n*a^(n-1)
        poly_der_coeff = np.flip(range(2, self.poly_coefficients.size))
        # evaluating the derivative polynomial n-coefficients: n-1*a^(n-2)
        poly_der2_coeff = np.flip(range(1, self.poly_coefficients.size - 1))
        coeff = self.poly_coefficients[:-2]

        # multiplying each polynomial variable term by its coefficient
        # evaluating variable powers for each term of the polynomial evaluated in the point provided
        variable_values = np.flip(
            [coords**c for c in range(self.poly_coefficients.size - 2)], axis=0
        )

        # evaluating variable powers for each term of the polynomial evaluated in the point provided
        # multiplying each polynomial variable term by its derivative coefficient and the polynomial coefficient
        return np.dot(poly_der_coeff * poly_der2_coeff * coeff, variable_values)


class SplineWrapper:
    """Spline wrapper implementation"""

    def __init__(self, axis: npt.ArrayLike, values: npt.ArrayLike) -> None:
        """Spline wrapper inputs.

        Parameters
        ----------
        axis : npt.ArrayLike
            axis array to build the spline interpolator
        values : npt.ArrayLike
            values of the spline at each point of the axis to build the spline interpolator
        """
        self.spline_interpolator = CubicSpline(np.asarray(axis), np.asarray(values))

    def evaluate(
        self, coords: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluating the interpolated spline at the specified variable values.

        Parameters
        ----------
        coords : Union[float, npt.ArrayLike]
            values of the variable at which evaluate the spline

        Returns
        -------
        Union[float, npt.ArrayLike]
            spline computed at each input variable value
        """
        coords = np.asarray(coords).astype("float64")

        return self.spline_interpolator(coords, 0, extrapolate=False)

    def evaluate_first_derivative(
        self, coords: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluating the interpolated spline first derivative at the specified variable values.

        Parameters
        ----------
        coords : Union[float, npt.ArrayLike]
            values of the variable at which evaluate the spline first derivative

        Returns
        -------
        Union[float, npt.ArrayLike]
            first derivative of interpolated spline computed at each input variable value
        """
        coords = np.asarray(coords).astype("float64")

        return self.spline_interpolator(coords, 1, extrapolate=False)

    def evaluate_second_derivative(
        self, coords: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluating the interpolated spline second derivative at the specified variable values.

        Parameters
        ----------
        coords : Union[float, npt.ArrayLike]
            values of the variable at which evaluate the spline second derivative

        Returns
        -------
        Union[float, npt.ArrayLike]
            second derivative of interpolated spline computed at each input variable value
        """
        coords = np.asarray(coords).astype("float64")

        return self.spline_interpolator(coords, 2, extrapolate=False)


class Generic3DCurve:
    """Implementation of a generic 3D curve from twice differentiable functions"""

    def __init__(
        self,
        x_func: RealTwiceDifferentiableFunction,
        y_func: RealTwiceDifferentiableFunction,
        z_func: RealTwiceDifferentiableFunction,
        t_ref: PreciseDateTime,
        time_boundaries: Tuple[PreciseDateTime, PreciseDateTime],
    ) -> None:
        """Generic twice differentiable 3D curve with a given domain of existence/interpolation.
        Must not extrapolate values outside of boundaries.

        Parameters
        ----------
        x_func : RealTwiceDifferentiableFunction
            twice differentiable function defining the curve behaviour along x, time definition is relative to t_ref
        y_func : RealTwiceDifferentiableFunction
            twice differentiable function defining the curve behaviour along y, time definition is relative to t_ref
        z_func : RealTwiceDifferentiableFunction
            twice differentiable function defining the curve behaviour along z, time definition is relative to t_ref
        t_ref : PreciseDateTime
            reference time of the 3D curve, aka the lower boundary of the maximum domain of definition
        time_boundaries : tuple[PreciseDateTime, PreciseDateTime]
            time boundaries of curve validity (domain)
        """
        self.x_func = x_func
        self.y_func = y_func
        self.z_func = z_func
        self.time_ref = t_ref
        # evaluating relative time boundaries
        self.time_boundaries = tuple(t - t_ref for t in time_boundaries)
        if not self.time_boundaries[1] > self.time_boundaries[0]:
            raise InvertedTimeBoundaries("Wrong time boundaries order")

    def evaluate(
        self, coordinates: Union[PreciseDateTime, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate x, y, z polynomial at given times.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, npt.ArrayLike]
            time points

        Returns
        -------
        np.ndarray
            values of the polynomials at given input times (N, 3)
        """

        # computing relative times
        times = np.asarray(coordinates)
        times = times - self.time_ref
        self._check_time_validity(relative_times=times)

        # evaluating polynomials at given times
        return np.stack(
            [
                self.x_func.evaluate(times),
                self.y_func.evaluate(times),
                self.z_func.evaluate(times),
            ],
            axis=-1,
        )

    def evaluate_first_derivatives(
        self, coordinates: Union[PreciseDateTime, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate x, y, z polynomial first derivatives at given times.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, npt.ArrayLike]
            time points

        Returns
        -------
        np.ndarray
            values of the polynomials first derivatives at given input times (N, 3)
        """

        # computing relative times
        times = np.asarray(coordinates)
        times = times - self.time_ref
        self._check_time_validity(relative_times=times)

        # evaluating polynomials at given times
        return np.stack(
            [
                self.x_func.evaluate_first_derivative(times),
                self.y_func.evaluate_first_derivative(times),
                self.z_func.evaluate_first_derivative(times),
            ],
            axis=-1,
        )

    def evaluate_second_derivatives(
        self, coordinates: Union[PreciseDateTime, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate x, y, z polynomial second derivatives at given times.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, npt.ArrayLike]
            time points

        Returns
        -------
        np.ndarray
            values of the polynomials second derivatives at given input times (N, 3)
        """

        # computing relative times
        times = np.asarray(coordinates)
        times = times - self.time_ref
        self._check_time_validity(relative_times=times)

        # evaluating polynomials at given times
        return np.stack(
            [
                self.x_func.evaluate_second_derivative(times),
                self.y_func.evaluate_second_derivative(times),
                self.z_func.evaluate_second_derivative(times),
            ],
            axis=-1,
        )

    def _check_time_validity(self, relative_times: Union[float, npt.ArrayLike]) -> None:
        """Check input times validity with respect to the time validity boundaries.

        Parameters
        ----------
        relative_times : Union[float, npt.ArrayLike]
            relative time points at which evaluate the polynomial

        Raises
        ------
        PolynomialInterpolationOutOfBoundaries
            if relative time is out of validity boundaries this error is raised (interpolation cannot be performed)
        """
        if ~np.any(
            (relative_times > self.time_boundaries[0])
            & (relative_times < self.time_boundaries[1])
        ):
            raise InterpolationOutOfBoundaries(
                "Interpolation failed. Times are outside domain"
            )
