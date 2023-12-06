# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Curve protocols definition
--------------------------
"""

from typing import Protocol, Union

import numpy as np
import numpy.typing as npt

from arepytools.timing.precisedatetime import PreciseDateTime


class TwiceDifferentiable3DCurve(Protocol):
    """Custom 3D curve container with evaluation method for curve and first two derivatives.
    f: R | PreciseDateTime -> R^3"""

    def evaluate(
        self, coordinates: Union[PreciseDateTime, float, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate curve value at given input coordinates.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, float, npt.ArrayLike]
            coordinates (PreciseDateTime) or float where to evaluate the curve

        Returns
        -------
        np.ndarray
            values of the curve at given input values (N, 3)
        """

    def evaluate_first_derivatives(
        self, coordinates: Union[PreciseDateTime, float, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate curve first derivatives values at given input coordinates.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, float, npt.ArrayLike]
            coordinates (PreciseDateTime) or float where to evaluate the derivatives

        Returns
        -------
        np.ndarray
            values of the curve derivatives at given input values (N, 3)
        """

    def evaluate_second_derivatives(
        self, coordinates: Union[PreciseDateTime, float, npt.ArrayLike]
    ) -> np.ndarray:
        """Evaluate curve second derivatives values at given input coordinates.

        Parameters
        ----------
        coordinates : Union[PreciseDateTime, float, npt.ArrayLike]
            coordinates (PreciseDateTime) or float where to evaluate the derivatives

        Returns
        -------
        np.ndarray
            values of the polynomials second derivatives at given input times (N, 3)
        """


class RealTwiceDifferentiableFunction(Protocol):
    """Generic protocol for a f: R -> R function twice differentiable with derivative evaluation methods implemented"""

    def evaluate(
        self, coordinates: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluate function value at given coordinates.

        Parameters
        ----------
        coordinates : Union[float, npt.ArrayLike]
            input coordinates where to evaluate the function

        Returns
        -------
        Union[float, npt.ArrayLike]
            value of function at each input coordinate
        """

    def evaluate_first_derivative(
        self, coordinates: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluate function first derivative at given coordinates.

        Parameters
        ----------
        coordinates : Union[float, npt.ArrayLike]
            input coordinates where to evaluate the function derivative

        Returns
        -------
        Union[float, npt.ArrayLike]
            values of function first derivative at each input coordinate
        """

    def evaluate_second_derivative(
        self, coordinates: Union[float, npt.ArrayLike]
    ) -> Union[float, npt.ArrayLike]:
        """Evaluate function second derivative at given coordinates.

        Parameters
        ----------
        coordinates : Union[float, npt.ArrayLike]
            input coordinates where to evaluate the function derivative

        Returns
        -------
        Union[float, npt.ArrayLike]
            values of function second derivative at each input coordinate
        """
