# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
ArePyTools: the Python toolkit for SAR data processing
"""

import logging
import sys

_MIN_PYTHON_VERSION = "3.8"

assert sys.version_info >= tuple(
    (int(v) for v in _MIN_PYTHON_VERSION.split("."))
), f"ArePyTools requires Python {_MIN_PYTHON_VERSION} or higher"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "1.6.1"
