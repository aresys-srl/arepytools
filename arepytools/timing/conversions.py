# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Date-Time conversions module
----------------------------
"""

from datetime import datetime
from typing import Tuple, Union

from arepytools.timing.precisedatetime import PreciseDateTime

_GPS_START_DATE = datetime(year=1980, month=1, day=6)


def date_to_gps_week(date: Union[PreciseDateTime, datetime]) -> Tuple[int, int]:
    """Convert input date to GPS week.
    GPS weeks are counted since 06-January-1980.

    Parameters
    ----------
    date : Union[PreciseDateTime, datetime]
        date to be converted to GPS week, in PreciseDateTime or datetime format

    Returns
    -------
    Tuple[int, int]
        GPS week
        GPS day of the week

    Raises
    ------
    ValueError
        if input date is before 06-January-1980
    """
    if isinstance(date, PreciseDateTime):
        # converting to datetime
        date = datetime(
            year=date.year,
            month=date.month,
            day=date.day_of_the_month,
            hour=date.hour_of_day,
            minute=date.minute_of_hour,
        )
    delta = date - _GPS_START_DATE

    if delta.days < 0:
        raise ValueError(f"Invalid date: {date} cannot be before {_GPS_START_DATE}")

    weeks = delta.days // 7
    day_of_week = delta.days % 7

    return weeks, day_of_week
