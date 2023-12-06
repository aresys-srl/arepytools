# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
**Deprecated** WGS84 module
---------------------------

.. deprecated:: v1.2.0
    Importing WGS84 elliposid from arepytools.geometry.wgs84 is deprecated.
    Use from :mod:`arepytools.geometry.ellipsoid` import :data:`WGS84`
"""

import warnings

from arepytools.geometry.ellipsoid import WGS84

warnings.warn(
    "arepytools.geometry.wgs84 module is deprecated: use 'from arepytools.geometry.ellipsoid import WGS84' instead",
    DeprecationWarning,
    stacklevel=2,
)
