# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Ellipsoid module
----------------
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Ellipsoid:
    """
    Ellipsoid class.

    Examples
    --------
    >>> ellipsoid = Ellipsoid(1., 3.)
    >>> ellipsoid = Ellipsoid(3., 1.)

    See also
    --------
    WGS84 : common earth ellispoid
    """

    semi_major_axis: float = field(init=False)
    semi_minor_axis: float = field(init=False)
    semi_axes_ratio_min_max: float = field(init=False)
    eccentricity_square: float = field(init=False)
    eccentricity: float = field(init=False)
    ep2: float = field(init=False)

    first_semi_axis: InitVar[float]
    second_semi_axis: InitVar[float]

    def __post_init__(self, first_semi_axis, second_semi_axis):
        """Initialize the object with the specified ellipsoid parameters

        Parameters
        ----------
        first_semi_axis : float
            length of the first semi-axis of the ellipsoid
        second_semi_axis : float
            length of the second semi-axis of the ellipsoid

        Raises
        ------
        ValueError
            in case of negative input axis
        """
        if first_semi_axis <= 0 or second_semi_axis <= 0:
            raise ValueError("Non-positive input axes")

        object.__setattr__(
            self, "semi_major_axis", max(first_semi_axis, second_semi_axis)
        )
        object.__setattr__(
            self, "semi_minor_axis", min(first_semi_axis, second_semi_axis)
        )
        object.__setattr__(
            self, "semi_axes_ratio_min_max", self.semi_minor_axis / self.semi_major_axis
        )
        object.__setattr__(
            self, "eccentricity_square", 1 - self.semi_axes_ratio_min_max**2
        )
        object.__setattr__(self, "eccentricity", np.sqrt(self.eccentricity_square))
        object.__setattr__(self, "ep2", 1.0 / self.semi_axes_ratio_min_max**2 - 1)

    def inflate(self, height: float) -> Ellipsoid:
        """Compute an ellipsoid by adding height to the semi-axis

        Parameters
        ----------
        height : float
            additional height, it can be negative

        Examples
        --------
        >>> a = Ellipsoid(1., 3.)
        >>> b = a.inflate(2)
        >>> print(b)
        Ellipsoid(semi_major_axis=5.0, semi_minor_axis=3.0,
                  semi_axes_ratio_min_max=0.6, eccentricity_square=0.64,
                  eccentricity=0.8, ep2=1.7777777777777777)
        """
        return Ellipsoid(self.semi_major_axis + height, self.semi_minor_axis + height)


_A_MAX = 6.378137e6  # semi-major axis
_A_MIN = 6.356752314245e6  # semi-minor axis

WGS84 = Ellipsoid(_A_MAX, _A_MIN)
"""WGS 84 ellipsoid, constant ellipsoid object

Examples
--------
>>> from arepytools.geometry.ellipsoid import WGS84
>>> print(WGS84.semi_minor_axis)
6356752.314245
>>> print(WGS84.semi_axes_ratio_min_max)
0.9966471893352243
"""


def compute_line_ellipsoid_intersections(
    line_directions: npt.ArrayLike, line_origins: npt.ArrayLike, ellipsoid: Ellipsoid
) -> Union[Tuple[Tuple[np.ndarray]], Tuple[np.ndarray]]:
    """Compute the intersections between lines and an ellipsoid

    For each line it returns the intersections.

    When two intersections are found they are sorted by the closest to line_origin.

    Parameters
    ----------
    line_directions : npt.ArrayLike
        (3,), (N, 3) one or more line directions, not necessarily normalized
    line_origins : npt.ArrayLike
        (3,), (N, 3) one or more line origins
    ellipsoid : Ellipsoid
        ellipsoid

    Returns
    -------
    Union[Tuple[Tuple[np.ndarray]], Tuple[np.ndarray]]
        The intersections or a tuple of intersections, depending on N.
        Intersections are stored as a tuple of points (np.array (3,)).
        The number of points depends on the intersection results can be 0, 1 or 2.

    Raises
    ------
    ValueError
        In case of invalid input

    Examples
    --------

    empty intersection

    >>> intersections = compute_line_ellipsoid_intersections(
                    [0, 0, 100], [-5, 0, 2], Ellipsoid(1, 2)
                )
    >>> print(intersections)
    ()

    single intersection

    >>> intersections = compute_line_ellipsoid_intersections(
                    [100, 0, 0], [-5, 0, 1], Ellipsoid(1, 2)
                )
    >>> print(intersections)
    (array([-8.8817842e-16,  0.0000000e+00,  1.0000000e+00]),)

    two intersections, first one is closer to line_origin

    >>> intersections = compute_line_ellipsoid_intersections(
                    [100, 0, 0], [-5, 0, 0], Ellipsoid(1, 2)
                )
    >>> print(intersections)
    (array([-2.,  0.,  0.]), array([2., 0., 0.]))

    multiple lines

    >>> line_origins = np.array([[-5, 0, 2], [-5, 0, 1], [-5, 0, 0]])
    >>> line_directions = np.array([[0, 0, 100], [100, 0, 0], [100, 0, 0]])
    >>> intersections = compute_line_ellipsoid_intersections(
                line_directions, line_origins, Ellipsoid(1, 2)
            )
    >>> print(intersections)
    ((), (array([-8.8817842e-16,  0.0000000e+00,  1.0000000e+00]),),
         (array([-2.,  0.,  0.]), array([2., 0., 0.])))
    """

    line_directions = np.asarray(line_directions)
    line_origins = np.asarray(line_origins)

    ndim = max(line_origins.ndim, line_directions.ndim)

    if line_directions.shape[-1] != 3:
        raise ValueError(
            f"Invalid line_direction shape: {line_directions.shape} should be (3,) or (N,3)"
        )

    if line_origins.shape[-1] != 3:
        raise ValueError(
            f"Invalid line_origin shape: {line_origins.shape} should be (3,) or (N,3)"
        )

    line_directions = line_directions / np.linalg.norm(
        line_directions, axis=-1, keepdims=True
    )

    # line: x = line_origin + t * line_direction
    # equation: t ** 2 + b t + c  = 0

    assert isinstance(line_directions, np.ndarray)
    line_directions_scaled = np.stack(
        [
            line_directions[..., 0] / ellipsoid.semi_major_axis,
            line_directions[..., 1] / ellipsoid.semi_major_axis,
            line_directions[..., 2] / ellipsoid.semi_minor_axis,
        ],
        axis=-1,
    )

    line_origins_scaled = np.stack(
        [
            line_origins[..., 0] / ellipsoid.semi_major_axis,
            line_origins[..., 1] / ellipsoid.semi_major_axis,
            line_origins[..., 2] / ellipsoid.semi_minor_axis,
        ],
        axis=-1,
    )

    num_lines = max(line_directions.size // 3, line_origins.size // 3)

    quadratic_terms = np.sum(line_directions_scaled * line_directions_scaled, axis=-1)
    linear_terms = 2 * np.sum(line_directions_scaled * line_origins_scaled, axis=-1)
    constant_terms = np.sum(line_origins_scaled * line_origins_scaled, axis=-1) - 1

    quadratic_terms = np.broadcast_to(quadratic_terms, (num_lines,))
    linear_terms = np.broadcast_to(linear_terms, (num_lines,))
    constant_terms = np.broadcast_to(constant_terms, (num_lines,))

    polynomials = (
        np.polynomial.Polynomial(coeffs)
        for coeffs in zip(constant_terms, linear_terms, quadratic_terms)
    )

    line_origins = np.broadcast_to(line_origins, (num_lines, 3))
    line_directions = np.broadcast_to(line_directions, (num_lines, 3))

    def solve_equation(poly: np.polynomial.Polynomial) -> tuple:
        return tuple(
            sorted(
                np.unique([root for root in poly.roots() if np.isreal(root)]), key=abs
            )
        )

    assert isinstance(line_directions, np.ndarray)
    assert isinstance(line_origins, np.ndarray)
    intersections_tuple = tuple(
        tuple(origin + root * direction for root in solve_equation(poly))
        for poly, origin, direction in zip(polynomials, line_origins, line_directions)
    )

    return intersections_tuple[0] if ndim == 1 else intersections_tuple
