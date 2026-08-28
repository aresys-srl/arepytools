# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""MetaData module.
-------------------
"""

from __future__ import annotations

import collections
import copy
import enum
import warnings
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt

from arepytools.timing.precisedatetime import PreciseDateTime

SECOND_STR = "s"
HERTZ_STR = "Hz"
JOULE_STR = "j"
RAD_STR = "rad"
UTC_STR = "Utc"


class EByteOrder(enum.Enum):
    """Byte orders supported ProductFolder's raster."""

    be = "BIGENDIAN"
    le = "LITTLEENDIAN"


class ECellType(enum.Enum):
    """Data format supported in the ProductFolder' raster."""

    int8 = "INT8"
    int16 = "INT16"
    int32 = "INT32"
    float32 = "FLOAT32"
    float64 = "FLOAT64"
    i8complex = "INT8_COMPLEX"
    i16complex = "INT16_COMPLEX"
    i32complex = "INT_COMPLEX"
    fcomplex = "FLOAT_COMPLEX"
    dcomplex = "DOUBLE_COMPLEX"
    custom = "CUSTOM"


class EOrbitDirection(enum.Enum):
    """Satellite orbit direction."""

    ascending = "ASCENDING"
    descending = "DESCENDING"


class ESideLooking(enum.Enum):
    """Satellite side looking."""

    right_looking = "RIGHT"
    left_looking = "LEFT"


class EPolarization(enum.Enum):
    """Polarizations."""

    hh = "H/H"
    vv = "V/V"
    hv = "H/V"
    vh = "V/H"
    none = None
    crh = "CR/H"
    crv = "CR/V"
    clh = "CL/H"
    clv = "CL/V"
    ch = "C/H"
    cv = "C/V"
    xh = "x/H"
    xv = "x/V"
    hx = "H/x"
    vx = "V/x"
    xx = "X/X"


class EPulseDirection(enum.Enum):
    """Chirp pulse type."""

    up = "UP"
    down = "DOWN"


class EReferenceFrame(enum.Enum):
    """Data reference frame."""

    geocentric = "GEOCENTRIC"
    geodetic = "GEODETIC"
    zerodoppler = "ZERODOPPLER"
    none = ""


class ERotationOrder(enum.Enum):
    """Attitude rotation order
    - y: yaw
    - p: pitch
    - r: roll.
    """

    ypr = "ypr"
    yrp = "yrp"
    pry = "pry"
    pyr = "pyr"
    ryp = "ryp"
    rpy = "rpy"
    none = ""


class EAttitudeType(enum.Enum):
    """Attitude type."""

    nominal = "NOMINAL"
    refined = "REFINED"
    none = ""


class ERasterFormatType(enum.Enum):
    """Aresys raster format."""

    aresys_raster = "ARESYS_RASTER"
    aresys_geotiff = "ARESYS_GEOTIFF"
    raster = "ARESYS_RASTER"


class MetaDataElement:
    """Base class for metadata elements."""

    def __repr__(self) -> str:
        max_len = max([len(x) for x in self.__dict__])
        str_repr = [f"\nMetaDataElement: {self.type()}\n\n"]
        str_repr += [
            "{elemName:>{length}}: {value}\n".format(elemName=k.lstrip("_.- "), length=max_len + 1, value=v)
            for k, v in self.__dict__.items()
        ]
        return "".join(str_repr)

    @classmethod
    def type(cls) -> str:
        """Return class name."""
        return cls.__name__

    def copy(self):
        return copy.copy(self)


class RasterInfo(MetaDataElement):
    """RasterInfo class.

    Parameters
    ----------
    lines : int
        The number of lines in the raster.
    samples : int
        The number of samples in the raster.
    celltype : Union[str, ECellType]
        The cell type of the raster.
    filename : Optional[str], optional
        The filename of the raster, by default None.
    header_offset_bytes : int, optional
        The offset in bytes for the header, by default 0.
    row_prefix_bytes : int, optional
        The number of prefix bytes per row, by default 0.
    byteorder : Union[str, EByteOrder], optional
        The byte order of the raster, by default "LITTLEENDIAN".
    invalid_value : Optional[Union[float, complex]], optional
        The invalid value of the raster, by default None.
    format_type : Union[ERasterFormatType, str, None], optional
        The format type of the raster, by default None.

    Attributes
    ----------
    file_name : Optional[str]
        The filename of the raster.
    lines : int
        The number of lines in the raster.
    samples : int
        The number of samples in the raster.
    header_offset_bytes : int
        The offset in bytes for the header.
    row_prefix_bytes : int
        The number of prefix bytes per row.
    lines_start : Union[float, PreciseDateTime]
        The start value of the lines axis.
    lines_start_date : PreciseDateTime
        The start value of the lines axis as a PreciseDateTime object.
    lines_start_unit : str
        The unit of the start value of the lines axis.
    lines_step : float
        The step value of the lines axis.
    lines_step_unit : str
        The unit of the step value of the lines axis.
    samples_start : Union[float, PreciseDateTime]
        The start value of the samples axis.
    samples_start_date : PreciseDateTime
        The start value of the samples axis as a PreciseDateTime object.
    samples_start_unit : str
        The unit of the start value of the samples axis.
    samples_step : float
        The step value of the samples axis.
    samples_step_unit : str
        The unit of the step value of the samples axis.
    byte_order : EByteOrder
        The byte order of the raster.
    cell_type : ECellType
        The cell type of the raster.
    invalid_value : Optional[Union[float, complex]]
        The invalid value of the raster.
    format_type : Optional[ERasterFormatType]
        The format type of the raster.

    Methods
    -------
    set_lines_axis(lines_start, lines_start_unit, lines_step, lines_step_unit)
        Set the lines axis parameters.
    set_samples_axis(samples_start, samples_start_unit, samples_step, samples_step_unit)
        Set the samples axis parameters.

    """

    def __init__(
        self,
        lines: int,
        samples: int,
        celltype: str | ECellType,
        filename: str | None = None,
        header_offset_bytes: int = 0,
        row_prefix_bytes: int = 0,
        byteorder: str | EByteOrder = "LITTLEENDIAN",
        invalid_value: complex | None = None,
        format_type: ERasterFormatType | str | None = None,
    ) -> None:
        self._file_name = filename

        self._lines = lines
        self._samples = samples
        self._header_offset_bytes = header_offset_bytes
        self._row_prefix_bytes = row_prefix_bytes

        self._byte_order = EByteOrder(byteorder)
        self._cell_type = ECellType(celltype)

        self._lines_start: float | PreciseDateTime = 0.0
        self._lines_start_unit = ""
        self._lines_step = 0.0
        self._lines_step_unit = ""

        self._samples_start: float | PreciseDateTime = 0.0
        self._samples_start_unit = ""
        self._samples_step = 0.0
        self._samples_step_unit = ""

        self._invalid_value = invalid_value

        if format_type is not None:
            format_type = ERasterFormatType(format_type)
        self._format_type = format_type

    @property
    def file_name(self) -> str | None:
        """Get the name of the file associated with this metadata.

        Returns
        -------
        Optional[str]
            The name of the file, or None if no file is associated.

        """
        return self._file_name

    @file_name.setter
    def file_name(self, filename: str) -> None:
        self._file_name = filename

    @property
    def lines(self) -> int:
        """Get the number of lines.

        Returns
        -------
        int
            The number of lines.

        """
        return self._lines

    @property
    def samples(self) -> int:
        """Get the number of samples.

        Returns
        -------
        int
            The number of samples.

        """
        return self._samples

    @property
    def header_offset_bytes(self) -> int:
        """Get the offset in bytes where the header ends.

        Returns
        -------
        int
            The offset in bytes where the header ends.

        """
        return self._header_offset_bytes

    @property
    def row_prefix_bytes(self) -> int:
        """Get the number of bytes used for the row prefix.

        Returns
        -------
        int
            The number of bytes used for the row prefix.

        """
        return self._row_prefix_bytes

    @property
    def lines_start(self) -> float | PreciseDateTime:
        """Get the start time of the lines.

        Returns
        -------
        Union[float, PreciseDateTime]
            The start time of the lines.

        """
        return self._lines_start

    @property
    def lines_start_date(self) -> PreciseDateTime:
        """Get the start date of the lines.

        Returns
        -------
        PreciseDateTime
            The start date of the lines.

        Raises
        ------
        RuntimeError
            If the lines start is not a PreciseDateTime.

        """
        if not isinstance(self._lines_start, PreciseDateTime):
            msg = "The lines start is not a PreciseDateTime"
            raise RuntimeError(msg)

        return self._lines_start

    @property
    def lines_start_unit(self) -> str:
        """Returns the unit of the lines start.

        Returns
        -------
        str
            The unit of the lines start.

        """
        return self._lines_start_unit

    @property
    def lines_step(self) -> float:
        """Returns the step value of the lines.

        Returns
        -------
        float
            The step value of the lines.

        """
        return self._lines_step

    @property
    def lines_step_unit(self) -> str:
        """Returns the unit of the lines step.

        Returns
        -------
        str
            The unit of the lines step.

        """
        return self._lines_step_unit

    @property
    def samples_start(self) -> float | PreciseDateTime:
        """Returns the start value of the samples.

        Returns
        -------
        Union[float, PreciseDateTime]
            The start value of the samples.

        """
        return self._samples_start

    @property
    def samples_start_date(self) -> PreciseDateTime:
        """Returns the start date of the samples.

        Returns
        -------
        PreciseDateTime
            The start date of the samples.

        Raises
        ------
        RuntimeError
            If the samples start is not a PreciseDateTime.

        """
        if not isinstance(self._samples_start, PreciseDateTime):
            msg = "The samples start is not a PreciseDateTime"
            raise RuntimeError(msg)

        return self._samples_start

    @property
    def samples_start_unit(self) -> str:
        """Returns the unit of the samples start.

        Returns
        -------
        str
            The unit of the samples start.

        """
        return self._samples_start_unit

    @property
    def samples_step(self) -> float:
        """Returns the step value of the samples.

        Returns
        -------
        float
            The step value of the samples.

        """
        return self._samples_step

    @property
    def samples_step_unit(self) -> str:
        """Returns the unit of the samples step.

        Returns
        -------
        str
            The unit of the samples step.

        """
        return self._samples_step_unit

    @property
    def byte_order(self) -> EByteOrder:
        """Returns the byte order of the raster data.

        Returns
        -------
        EByteOrder
            The byte order of the raster data.

        """
        return self._byte_order

    @property
    def cell_type(self) -> ECellType:
        """Returns the cell type of the raster data.

        Returns
        -------
        ECellType
            The cell type of the raster data.

        """
        return self._cell_type

    @property
    def invalid_value(self) -> float | complex | None:
        """Returns the invalid value for the raster data.

        Returns
        -------
        Optional[Union[float, complex]]
            The invalid value for the raster data, or None if not set.

        """
        return self._invalid_value

    @property
    def format_type(self) -> ERasterFormatType | None:
        """Returns the format type of the raster data.

        Returns
        -------
        Optional[ERasterFormatType]
            The format type of the raster data, or None if not set.

        """
        return self._format_type

    def set_lines_axis(
        self,
        lines_start: float | PreciseDateTime,
        lines_start_unit: str,
        lines_step: float,
        lines_step_unit: str,
    ) -> None:
        """Set the lines axis parameters.

        Parameters
        ----------
        lines_start : Union[float, PreciseDateTime]
            The start value of the lines axis.
        lines_start_unit : str
            The unit of the start value of the lines axis.
        lines_step : float
            The step value of the lines axis.
        lines_step_unit : str
            The unit of the step value of the lines axis.

        """
        self._lines_start = lines_start
        self._lines_start_unit = lines_start_unit
        self._lines_step = lines_step
        self._lines_step_unit = lines_step_unit

    def set_samples_axis(
        self,
        samples_start: float | PreciseDateTime,
        samples_start_unit: str,
        samples_step: float,
        samples_step_unit: str,
    ) -> None:
        """Set the samples axis parameters.

        Parameters
        ----------
        samples_start : Union[float, PreciseDateTime]
            The start value of the samples axis.
        samples_start_unit : str
            The unit of the start value of the samples axis.
        samples_step : float
            The step value of the samples axis.
        samples_step_unit : str
            The unit of the step value of the samples axis.

        """
        self._samples_start = samples_start
        self._samples_start_unit = samples_start_unit
        self._samples_step = samples_step
        self._samples_step_unit = samples_step_unit


class ImageQuantity(enum.Enum):
    """Image quantity."""

    BETA = "BETA"
    SIGMA = "SIGMA"
    GAMMA = "GAMMA"


class DataSetInfo(MetaDataElement):
    """DataSetInfo class."""

    def __init__(
        self,
        acquisition_mode_i: str | None = None,
        fc_hz_i: float | None = None,
        image_quantity: ImageQuantity | Literal["BETA", "SIGMA", "GAMMA"] | None = None,
    ) -> None:
        self.sensor_name: str | None = None
        self.description: str | None = None
        self.sense_date: PreciseDateTime | None = None
        self.acquisition_mode: str | None = acquisition_mode_i
        self.image_type: str | None = None
        self.projection: str | None = None
        self.acquisition_station: str | None = None
        self.processing_center: str | None = None
        self.processing_date: PreciseDateTime | None = None
        self.processing_software: str | None = None
        self.fc_hz = fc_hz_i
        self._side_looking = None
        self.external_calibration_factor: float | None = None
        self.data_take_id: int | None = None
        self.image_quantity = ImageQuantity(image_quantity) if image_quantity is not None else None
        self.projection_params: str | None = None
        self.projection_params_format: str | None = None
        self.instrument_conf_id: int | None = None

    @property
    def side_looking(self) -> ESideLooking | None:
        return self._side_looking

    @side_looking.setter
    def side_looking(self, side_looking_i: ESideLooking | None) -> None:
        self._side_looking = ESideLooking(side_looking_i)


@dataclass
class GeoPoint(MetaDataElement):
    """GeoPoint class."""

    lat: float = 0.0
    lon: float = 0.0
    height: float = 0.0
    theta_inc: float = 0.0
    theta_look: float = 0.0

    def to_list(self) -> list[float]:
        """Retrieve the geo point as a list:  [lat, lon, height, theta_inc, theta_look]."""
        return [self.lat, self.lon, self.height, self.theta_inc, self.theta_look]


@dataclass
class GroundCornerPoints(MetaDataElement):
    """GroundCornerPoint class."""

    easting_grid_size: float = 0.0
    northing_grid_size: float = 0.0
    center_point: GeoPoint = field(default_factory=GeoPoint)
    ne_point: GeoPoint = field(default_factory=GeoPoint)
    nw_point: GeoPoint = field(default_factory=GeoPoint)
    se_point: GeoPoint = field(default_factory=GeoPoint)
    sw_point: GeoPoint = field(default_factory=GeoPoint)

    @property
    def geo_points(self) -> list[GeoPoint]:
        """Return geo points as a list:  [nw_point, ne_point, sw_point, se_point, center_point]."""
        return [self.nw_point, self.ne_point, self.sw_point, self.se_point, self.center_point]


class SwathInfo(MetaDataElement):
    """SwathInfo class."""

    def __init__(
        self,
        swath_i: str | None = None,
        polarization_i: EPolarization | str | None = None,
        acquisition_prf_i: float = 0.0,
    ) -> None:
        self.swath: str | None = swath_i
        self.polarization = polarization_i
        self.acquisition_prf = acquisition_prf_i
        self.acquisition_prf_unit = HERTZ_STR
        self.swath_acquisition_order = 0
        self.rank = 0
        self.range_delay_bias = 0.0
        self.range_delay_bias_unit = SECOND_STR
        self.acquisition_start_time = None
        self.acquisition_start_time_unit = UTC_STR
        self._azimuth_steering_rate_reference_time: float | None = 0.0
        self._azimuth_steering_angle_reference_time: float | None = None
        self.az_steering_rate_ref_time_unit = SECOND_STR
        self.az_steering_angle_ref_time_unit = SECOND_STR
        self.echoes_per_burst = 0
        self._azimuth_steering_rate_pol: tuple[float, float, float] | None = (
            0.0,
            0.0,
            0.0,
        )
        self._azimuth_steering_angle_pol: tuple[float, float, float, float] | None = None
        self.rx_gain: float | None = None
        self.channel_delay: float | None = None

    @property
    def polarization(self) -> EPolarization:
        return self._polarization

    @polarization.setter
    def polarization(self, polarization_i: EPolarization | str | None) -> None:
        self._polarization = EPolarization(polarization_i)

    @property
    def acquisition_prf(self) -> float:
        return self._acquisition_prf

    @acquisition_prf.setter
    def acquisition_prf(self, acquisition_prf_i: float) -> None:
        self._acquisition_prf = acquisition_prf_i

    @property
    def acquisition_start_time(self) -> PreciseDateTime | None:
        return self._acquisition_start_time

    @acquisition_start_time.setter
    def acquisition_start_time(self, acquisition_start_time_i: PreciseDateTime | None) -> None:
        if isinstance(acquisition_start_time_i, PreciseDateTime):
            self._acquisition_start_time = acquisition_start_time_i
        elif acquisition_start_time_i is None:
            self._acquisition_start_time = None
        else:
            msg = "Acquisition start time has to be a PreciseDateTime"
            raise TypeError(msg)

    @property
    def azimuth_steering_rate_reference_time(self) -> float | None:
        return self._azimuth_steering_rate_reference_time

    @azimuth_steering_rate_reference_time.setter
    def azimuth_steering_rate_reference_time(self, i_azimuth_steering_rate_reference_time: float | None) -> None:
        self._azimuth_steering_rate_reference_time = i_azimuth_steering_rate_reference_time
        if i_azimuth_steering_rate_reference_time is not None:
            # forcing the other to None
            self._azimuth_steering_angle_reference_time = None

    @property
    def azimuth_steering_angle_reference_time(self) -> float | None:
        return self._azimuth_steering_angle_reference_time

    @azimuth_steering_angle_reference_time.setter
    def azimuth_steering_angle_reference_time(
        self,
        i_azimuth_steering_angle_reference_time: float | None,
    ) -> None:
        self._azimuth_steering_angle_reference_time = i_azimuth_steering_angle_reference_time
        if i_azimuth_steering_angle_reference_time is not None:
            # forcing the other to None
            self._azimuth_steering_rate_reference_time = None

    @property
    def azimuth_steering_rate_pol(self) -> tuple[float, float, float] | None:
        return self._azimuth_steering_rate_pol

    @azimuth_steering_rate_pol.setter
    def azimuth_steering_rate_pol(self, i_azimuth_steering_rate_pol: tuple[float, float, float] | None) -> None:
        if i_azimuth_steering_rate_pol is not None:
            if len(i_azimuth_steering_rate_pol) == 3 and isinstance(i_azimuth_steering_rate_pol, tuple):
                self._azimuth_steering_rate_pol = i_azimuth_steering_rate_pol
                # forcing the other to None
                self._azimuth_steering_angle_pol = None
            else:
                msg = "The azimuth steering rate pol has to be a tuple of 3 elements or None"
                raise TypeError(msg)
        else:
            self._azimuth_steering_rate_pol = None

    @property
    def azimuth_steering_angle_pol(
        self,
    ) -> tuple[float, float, float, float] | None:
        return self._azimuth_steering_angle_pol

    @azimuth_steering_angle_pol.setter
    def azimuth_steering_angle_pol(
        self,
        i_azimuth_steering_angle_pol: tuple[float, float, float, float] | None,
    ) -> None:
        if i_azimuth_steering_angle_pol is not None:
            if len(i_azimuth_steering_angle_pol) == 4 and isinstance(i_azimuth_steering_angle_pol, tuple):
                self._azimuth_steering_angle_pol = i_azimuth_steering_angle_pol
                # forcing the other to None
                self._azimuth_steering_rate_pol = None
            else:
                msg = "The azimuth steering pol has to be a tuple of 4 elements or None"
                raise TypeError(msg)
        else:
            self._azimuth_steering_angle_pol = None


class SamplingConstants(MetaDataElement):
    """SamplingConstants class."""

    def __init__(
        self,
        frg_hz_i: float | None = None,
        brg_hz_i: float | None = None,
        faz_hz_i: float | None = None,
        baz_hz_i: float | None = None,
    ) -> None:
        self.frg_hz = frg_hz_i
        self.frg_hz_unit = HERTZ_STR
        self.brg_hz = brg_hz_i
        self.brg_hz_unit = HERTZ_STR
        self.faz_hz = faz_hz_i
        self.faz_hz_unit = HERTZ_STR
        self.baz_hz = baz_hz_i
        self.baz_hz_unit = HERTZ_STR


class AcquisitionTimeLine(MetaDataElement):
    """AcquisitionTimeLine class."""

    def __init__(
        self,
        missing_lines_number_i: int = 0,
        missing_lines_azimuth_times_i: list[float] | None = None,
        swst_changes_number_i: int = 0,
        swst_changes_azimuth_times_i: list[float] | None = None,
        swst_changes_values_i: list[float] | None = None,
        noise_packets_number_i: int = 0,
        noise_packets_azimuth_times_i: list[float] | None = None,
        internal_calibration_number_i: int = 0,
        internal_calibration_azimuth_times_i: list[float] | None = None,
        swl_changes_number_i: int = 0,
        swl_changes_azimuth_times_i: list[float] | None = None,
        swl_changes_values_i: list[float] | None = None,
        prf_changes_number_i: int = 0,
        prf_changes_azimuth_times_i: list[float] | None = None,
        prf_changes_values_i: list[float] | None = None,
        chirp_period: str | None = None,
    ) -> None:
        inputs_to_validate = {
            "missing_lines_number_i": (
                missing_lines_number_i,
                [missing_lines_azimuth_times_i],
            ),
            "swst_changes_number_i": (
                swst_changes_number_i,
                [swst_changes_azimuth_times_i, swst_changes_values_i],
            ),
            "noise_packets_number_i": (
                noise_packets_number_i,
                [noise_packets_azimuth_times_i],
            ),
            "internal_calibration_number_i": (
                internal_calibration_number_i,
                [internal_calibration_azimuth_times_i],
            ),
            "swl_changes_number_i": (
                swl_changes_number_i,
                [swl_changes_azimuth_times_i, swl_changes_values_i],
            ),
            "prf_changes_number_i": (
                prf_changes_number_i,
                [prf_changes_azimuth_times_i, prf_changes_values_i],
            ),
        }
        for tag, (number, list_of_vec) in inputs_to_validate.items():
            for vec in list_of_vec:
                if vec is not None and len(vec) != number:
                    msg = f"Incorrect size of vectors ({tag}) {len(vec)} != {number}"
                    raise ValueError(msg)

        self._missing_lines_number = missing_lines_number_i
        self._missing_lines_azimuth_times = missing_lines_azimuth_times_i

        self._duplicated_lines_number = 0
        self._duplicated_lines_azimuth_times = None

        self._swst_changes_number = swst_changes_number_i
        self._swst_changes_azimuth_times = swst_changes_azimuth_times_i
        self._swst_changes_values = swst_changes_values_i
        self._noise_packets_number = noise_packets_number_i
        self._noise_packets_azimuth_times = noise_packets_azimuth_times_i
        self._internal_calibration_number = internal_calibration_number_i
        self._internal_calibration_azimuth_times = internal_calibration_azimuth_times_i
        self._swl_changes_number = swl_changes_number_i
        self._swl_changes_azimuth_times = swl_changes_azimuth_times_i
        self._swl_changes_values = swl_changes_values_i
        self._prf_changes_number = prf_changes_number_i
        self._prf_changes_azimuth_times = prf_changes_azimuth_times_i
        self._prf_changes_values = prf_changes_values_i

        # Units:
        self.missing_lines_azimuth_times_unit = SECOND_STR
        self.duplicated_lines_azimuth_times_unit = SECOND_STR
        self.swst_changes_azimuth_times_unit = SECOND_STR
        self.swst_changes_values_unit = SECOND_STR
        self.noise_packets_azimuth_times_unit = SECOND_STR
        self.internal_calibration_azimuth_times_unit = SECOND_STR
        self.swl_changes_azimuth_times_unit = SECOND_STR
        self.swl_changes_values_unit = SECOND_STR
        self.prf_changes_azimuth_times_unit = SECOND_STR
        self.prf_changes_values_unit = HERTZ_STR

        self.chirp_period = chirp_period

    @property
    def duplicated_lines(self) -> tuple[int, list[float] | None]:
        return self._duplicated_lines_number, self._duplicated_lines_azimuth_times

    @duplicated_lines.setter
    def duplicated_lines(self, duplicated_lines: tuple[int, list[float]]) -> None:
        if len(duplicated_lines[1]) != duplicated_lines[0]:
            msg = f"Duplicated lines inconsistent tuple: {duplicated_lines[0]} != {len(duplicated_lines[1])}"
            raise ValueError(
                msg,
            )

        self._duplicated_lines_number = duplicated_lines[0]
        self._duplicated_lines_azimuth_times = duplicated_lines[1]

    @property
    def internal_calibration(self) -> tuple[int, list[float] | None]:
        return (
            self._internal_calibration_number,
            self._internal_calibration_azimuth_times,
        )

    @internal_calibration.setter
    def internal_calibration(self, internal_calibration: list[float]) -> None:
        self._internal_calibration_number = len(internal_calibration)
        self._internal_calibration_azimuth_times = internal_calibration

    @property
    def missing_lines(self) -> tuple[int, list[float] | None]:
        return self._missing_lines_number, self._missing_lines_azimuth_times

    @missing_lines.setter
    def missing_lines(self, azimuth_times: list[float]) -> None:
        self._missing_lines_number = len(azimuth_times)
        self._missing_lines_azimuth_times = azimuth_times

    @property
    def noise_packet(self) -> tuple[int, list[float] | None]:
        return self._noise_packets_number, self._noise_packets_azimuth_times

    @noise_packet.setter
    def noise_packet(self, azimuth_times: list[float]) -> None:
        self._noise_packets_number = len(azimuth_times)
        self._noise_packets_azimuth_times = azimuth_times

    @property
    def swl_changes(self) -> tuple[int, list[float] | None, list[float] | None]:
        return (
            self._swl_changes_number,
            self._swl_changes_azimuth_times,
            self._swl_changes_values,
        )

    @swl_changes.setter
    def swl_changes(self, swl_changes: tuple[int, list[float], list[float]]) -> None:
        if swl_changes[0] != len(swl_changes[1]):
            msg = "Inconsistent swl changes sizes"
            raise ValueError(msg)
        self._swl_changes_number = swl_changes[0]
        self._swl_changes_azimuth_times = swl_changes[1]
        self._swl_changes_values = swl_changes[2]

    @property
    def swst_changes(self) -> tuple[int, list[float] | None, list[float] | None]:
        return (
            self._swst_changes_number,
            self._swst_changes_azimuth_times,
            self._swst_changes_values,
        )

    @swst_changes.setter
    def swst_changes(self, swst_changes: tuple[int, list[float], list[float]]) -> None:
        if swst_changes[0] != len(swst_changes[1]):
            msg = "Inconsistent swst changes sizes"
            raise ValueError(msg)
        self._swst_changes_number = swst_changes[0]
        self._swst_changes_azimuth_times = swst_changes[1]
        self._swst_changes_values = swst_changes[2]

    @property
    def prf_changes(self) -> tuple[int, list[float] | None, list[float] | None]:
        return (
            self._prf_changes_number,
            self._prf_changes_azimuth_times,
            self._prf_changes_values,
        )

    @prf_changes.setter
    def prf_changes(self, prf_changes: tuple[int, list[float], list[float]]) -> None:
        if prf_changes[0] != len(prf_changes[1]):
            msg = "Inconsistent prf changes sizes"
            raise ValueError(msg)
        self._prf_changes_number = prf_changes[0]
        self._prf_changes_azimuth_times = prf_changes[1]
        self._prf_changes_values = prf_changes[2]


class AttitudeInfo(MetaDataElement):
    """AttitudeInfo class."""

    _default_attitude_type = EAttitudeType("NOMINAL")

    def __init__(
        self,
        yaw: npt.ArrayLike | None = None,
        pitch: npt.ArrayLike | None = None,
        roll: npt.ArrayLike | None = None,
        t0: PreciseDateTime | None = None,
        delta_t: float = 0.0,
        ref_frame: str | None = None,
        rot_order: str | None = None,
    ) -> None:
        if ref_frame is None:
            ref_frame = ""
        if rot_order is None:
            rot_order = ""
        if t0 is None:
            t0 = PreciseDateTime()

        self._reset_angles(yaw, pitch, roll)

        self.reference_frame = ref_frame
        self.rotation_order = rot_order

        self._attitude_type = self._default_attitude_type

        self._t_ref_Utc = t0
        self._dtYPR_s = delta_t

    def _reset_angles(
        self,
        yaw: npt.ArrayLike | None,
        pitch: npt.ArrayLike | None,
        roll: npt.ArrayLike | None,
    ) -> None:
        if yaw is None and pitch is None and roll is None:
            return
        yaw_vector = np.array(yaw)
        pitch_vector = np.array(pitch)
        roll_vector = np.array(roll)

        yaw_size = yaw_vector.size
        for vec, tag in zip((yaw_vector, pitch_vector, roll_vector), ("yaw", "pitch", "roll")):
            wrong_dimension_string = f"Provided {tag} vector shall be a 1xN or Nx1 array"
            if vec.ndim == 1:
                if vec.size != yaw_size:
                    raise ValueError(wrong_dimension_string)
            elif vec.ndim == 2:
                if vec.shape[0] != 1 and vec.shape[1] != 1:
                    raise ValueError(wrong_dimension_string)
            else:
                raise ValueError(wrong_dimension_string)

            if yaw_size != vec.size:
                msg = f"{tag} and yaw vectors must have the same number of elements"
                raise ValueError(msg)
        self._yaw_deg = yaw_vector
        self._pitch_deg = pitch_vector
        self._roll_deg = roll_vector
        self._nYPR_n = yaw_size

    @property
    def reference_frame(self) -> EReferenceFrame:
        return self._reference_frame

    @reference_frame.setter
    def reference_frame(self, reference_frame: str) -> None:
        self._reference_frame = EReferenceFrame(reference_frame.upper())

    @property
    def rotation_order(self) -> ERotationOrder:
        return self._rotation_order

    @rotation_order.setter
    def rotation_order(self, rotation_order: str) -> None:
        self._rotation_order = ERotationOrder(rotation_order.lower())

    @property
    def attitude_records_number(self) -> int:
        return self._nYPR_n

    @property
    def attitude_type(self) -> EAttitudeType:
        return self._attitude_type

    @attitude_type.setter
    def attitude_type(self, attitude_type: str) -> None:
        self._attitude_type = EAttitudeType(attitude_type.upper())

    @property
    def pitch_vector(self) -> np.ndarray:
        return self._pitch_deg

    @property
    def reference_time(self) -> int | float | PreciseDateTime | None:
        return self._t_ref_Utc

    @reference_time.setter
    def reference_time(self, reference_time: float | PreciseDateTime) -> None:
        if not isinstance(reference_time, (int, float, PreciseDateTime)):
            msg = "Input start time must be a scalar PreciseDateTime"
            raise ValueError(msg)
        self._t_ref_Utc = reference_time

    @property
    def roll_vector(self) -> np.ndarray:
        return self._roll_deg

    @property
    def time_step(self) -> float:
        return self._dtYPR_s

    @time_step.setter
    def time_step(self, delta_t: float) -> None:
        self._dtYPR_s = delta_t

    @property
    def yaw_vector(self) -> np.ndarray:
        return self._yaw_deg

    def set_attitude_angles_vectors(
        self,
        yaw: npt.ArrayLike | None,
        pitch: npt.ArrayLike | None,
        roll: npt.ArrayLike | None,
    ) -> None:
        self._reset_angles(yaw, pitch, roll)


class Burst(MetaDataElement):
    """Single Burst class."""

    def __init__(
        self,
        range_start_time_i: float,
        azimuth_start_time_i: PreciseDateTime,
        lines_i: int,
        burst_center_azimuth_shift_i: float | None = None,
    ) -> None:
        self._range_start_time = range_start_time_i
        self._azimuth_start_time = azimuth_start_time_i
        self._burst_center_azimuth_shift = burst_center_azimuth_shift_i
        self._lines = lines_i

    @property
    def range_start_time(self) -> float:
        return self._range_start_time

    @property
    def azimuth_start_time(self) -> PreciseDateTime:
        return self._azimuth_start_time

    @property
    def burst_center_azimuth_shift(self) -> float | None:
        return self._burst_center_azimuth_shift

    @property
    def lines(self) -> int:
        return self._lines


class BurstInfo(MetaDataElement):
    """BurstInfo class."""

    def __init__(self, burst_repetition_frequency: float = 0.0) -> None:
        self._bursts: list[Burst] = []
        self._lines_per_burst_present = False
        self._lines_per_burst = 0
        self.burst_repetition_frequency = burst_repetition_frequency

    def is_lines_per_burst_present(self) -> bool:
        return self._lines_per_burst_present

    @property
    def lines_per_burst(self) -> int:
        return self._lines_per_burst

    def add_burst(
        self,
        range_start_time_i: float,
        azimuth_start_time_i: PreciseDateTime,
        lines_i: int,
        burst_center_azimuth_shift_i: float | None = None,
    ) -> None:
        if len(self._bursts) == 0:
            self._lines_per_burst_present = True
            self._lines_per_burst = lines_i
        elif self._lines_per_burst_present:
            if self._lines_per_burst != lines_i:
                self._lines_per_burst_present = False

        self._bursts.append(
            Burst(
                range_start_time_i,
                azimuth_start_time_i,
                lines_i,
                burst_center_azimuth_shift_i,
            ),
        )

    def get_number_of_bursts(self) -> int:
        return len(self._bursts)

    def get_burst(self, burst_index: int) -> Burst:
        if 0 <= burst_index < self.get_number_of_bursts():
            return self._bursts[burst_index]

        msg = "Burst index out of range"
        raise RuntimeError(msg)

    def clear_bursts(self) -> None:
        self._bursts = []

    def _raise_on_invalid_burst_index(self, burst_index: int) -> None:
        if burst_index < 0 or burst_index >= self.get_number_of_bursts():
            msg = "Not valid burst index"
            raise ValueError(msg)

    def get_lines(self, burst_index: int | None = None) -> np.ndarray | int:
        if burst_index is None:
            return np.asarray([burst.lines for burst in self._bursts])
        self._raise_on_invalid_burst_index(burst_index)
        return self._bursts[burst_index].lines

    def get_azimuth_start_time(self, burst_index: int | None = None) -> np.ndarray | PreciseDateTime:
        if burst_index is None:
            return np.asarray([burst.azimuth_start_time for burst in self._bursts])
        self._raise_on_invalid_burst_index(burst_index)
        return self._bursts[burst_index].azimuth_start_time

    def get_burst_center_azimuth_shift(self, burst_index: int | None = None) -> np.ndarray | float | None:
        if burst_index is None:
            return np.asarray([burst.burst_center_azimuth_shift for burst in self._bursts])
        self._raise_on_invalid_burst_index(burst_index)
        return self._bursts[burst_index].burst_center_azimuth_shift

    def get_range_start_time(self, burst_index: int | None = None) -> np.ndarray | float:
        if burst_index is None:
            return np.asarray([burst.range_start_time for burst in self._bursts])
        self._raise_on_invalid_burst_index(burst_index)
        return self._bursts[burst_index].range_start_time

    def get_burst_roi(self, burst_index: int, raster_info: RasterInfo, roi_range: list[int] | None = None) -> list[int]:
        if roi_range is None:
            roi_range = [0, raster_info.samples]

        accumulated_lines = 0
        for i_burst in range(burst_index):
            accumulated_lines += self.get_lines(i_burst)

        first_line = accumulated_lines
        return [first_line, roi_range[0], self.get_lines(burst_index), roi_range[1]]  # type: ignore


class StateVectors(MetaDataElement):
    """StateVectors class."""

    def __init__(
        self,
        position_vector: np.ndarray | None = None,
        velocity_vector: np.ndarray | None = None,
        t_ref_utc: PreciseDateTime | None = None,
        dt_sv_s: float = 0.0,
    ) -> None:
        position_vector = np.array(position_vector)
        velocity_vector = np.array(velocity_vector)

        if position_vector.ndim > 2 or position_vector.size % 3 != 0:
            msg = "Wrong array size for input position vector"
            raise ValueError(msg)
        if velocity_vector.ndim > 2 or velocity_vector.size % 3 != 0:
            msg = "Wrong array size for input velocity vector"
            raise ValueError(msg)

        self._position_vector = position_vector
        self._velocity_vector = velocity_vector
        self._t_ref_utc = t_ref_utc
        self._dt_sv_s = dt_sv_s
        self._orbit_number = -1
        self._track_number = -1
        self._anx_time = None
        self._anx_position = None

    @property
    def anx_position(self) -> list[float] | None:
        """Anx position."""
        return self._anx_position

    def get_anx_time(self) -> PreciseDateTime | None:
        """.. deprecated:: v1.1.0
        Use :data:`anx_time` property instead.
        """
        warnings.warn(
            "get_anx_time is deprecated: use anx_time instead",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.anx_time

    @property
    def anx_time(self) -> PreciseDateTime | None:
        """Anx time."""
        return self._anx_time

    @property
    def orbit_direction(self) -> EOrbitDirection:
        """Orbit direction."""
        if self._velocity_vector[0][2] > 0:
            return EOrbitDirection.ascending

        return EOrbitDirection.descending

    @property
    def orbit_number(self) -> int:
        """Orbit number."""
        return self._orbit_number

    @property
    def position_vector(self) -> np.ndarray | None:
        """Position state vectors."""
        return self._position_vector

    @property
    def reference_time(self) -> PreciseDateTime | None:
        """Reference time."""
        return self._t_ref_utc

    @property
    def number_of_state_vectors(self) -> int:
        """Number of state vectors."""
        return int(self._position_vector.size / 3)

    @property
    def time_step(self) -> float:
        """Time step."""
        return self._dt_sv_s

    @property
    def track_number(self) -> int:
        """Track number."""
        return self._track_number

    @property
    def velocity_vector(self) -> np.ndarray:
        """Velocity state vectors."""
        return self._velocity_vector

    def set_axn_info(self, i_anx_time: PreciseDateTime, i_anx_pos: list[float]) -> None:
        """.. deprecated:: v1.1.0
        use :func:`set_anx_info` instead.
        """
        warnings.warn(
            "set_axn_info is deprecated: use set_anx_info instead",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.set_anx_info(i_anx_time, i_anx_pos)

    def set_anx_info(self, i_anx_time: PreciseDateTime | None, i_anx_pos: list[float] | None) -> None:
        """Set anx time and position."""
        assert (self._anx_time is not None and self._anx_position is not None) or (
            self._anx_time is None and self._anx_position is None
        )

        if (i_anx_time is None and i_anx_pos is not None) or (i_anx_time is not None and i_anx_pos is None):
            if i_anx_time is not None:
                msg = "It is not allowed to specify ANX time only"
                raise ValueError(msg)

            assert i_anx_pos is not None
            msg = "It is not allowed to specify ANX position only"
            raise ValueError(msg)

        assert (i_anx_time is not None and i_anx_pos is not None) or (i_anx_time is None and i_anx_pos is None)

        if i_anx_pos is not None and (not isinstance(i_anx_pos, list) or len(i_anx_pos) != 3):
            msg = "Wrong input type for ANX position. It must be None or a list of 3 elements"
            raise TypeError(msg)

        if i_anx_time is not None and not isinstance(i_anx_time, PreciseDateTime):
            msg = "Wrong input type for ANX time. It must be None or a PreciseDateTime object"
            raise TypeError(msg)

        self._anx_position = i_anx_pos
        self._anx_time = i_anx_time

    @orbit_number.setter
    def orbit_number(self, i_orbit_number: int) -> None:
        if i_orbit_number <= 0 or not isinstance(i_orbit_number, int):
            msg = "Provided orbit number must have an integer and positive value."
            raise ValueError(msg)
        self._orbit_number = i_orbit_number

    @track_number.setter
    def track_number(self, i_track_number: int) -> None:
        if not isinstance(i_track_number, int) or i_track_number <= 0:
            msg = "Provided track number must have an integer and positive value."
            raise ValueError(msg)
        self._track_number = i_track_number


class _Poly2D(MetaDataElement):
    """Base class for Poly2D format."""

    _POWERS_X = (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0)

    _POWERS_Y = (0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8)

    _UNITS = 10 * ("",)

    def __init__(
        self,
        i_ref_az: PreciseDateTime | float | None = None,
        i_ref_rg: float | None = None,
        i_coefficients: list[float] | None = None,
    ) -> None:
        self.t_ref_az = i_ref_az
        self.t_ref_rg = i_ref_rg
        i_coefficients = i_coefficients or []
        if len(i_coefficients) > len(self._POWERS_X):
            msg = "the size of coefficients and powers must agree"
            raise ValueError(msg)
        self._coefficients = i_coefficients

    @property
    def coefficients(self) -> list[float]:
        return self._coefficients

    @staticmethod
    def get_powers_x() -> tuple[int, ...]:
        return _Poly2D._POWERS_X

    @staticmethod
    def get_powers_y() -> tuple[int, ...]:
        return _Poly2D._POWERS_Y

    @classmethod
    def get_units(cls) -> tuple[str, ...]:
        return cls._UNITS


class _Poly2DVector(MetaDataElement):
    """Base class for list of Poly2D."""

    _SINGLE_POLY_TYPE = _Poly2D

    def __init__(self, i_poly2d: list | None = None) -> None:
        self._poly_list = i_poly2d if i_poly2d is not None else []
        assert isinstance(self._poly_list, list), "The input should be a list"
        self._current_poly_index = 0

    def __iter__(self) -> _Poly2DVector:
        return self

    def __next__(self) -> _Poly2D:
        if self._current_poly_index >= self.get_number_of_poly():
            self._current_poly_index = 0
            raise StopIteration
        self._current_poly_index += 1
        return self.get_poly(self._current_poly_index - 1)

    def __len__(self) -> int:
        return len(self._poly_list)

    def add_poly(self, i_poly2d: _Poly2D) -> None:
        self._poly_list.append(i_poly2d)

    def get_number_of_poly(self) -> int:
        return len(self._poly_list)

    def get_poly(self, index: int) -> _Poly2D:
        if index < 0 or index >= self.get_number_of_poly():
            msg = f"Polynomial not available for index: {index}"
            raise IndexError(msg)

        return self._poly_list[index]

    @classmethod
    def get_single_poly_type(cls) -> type[_Poly2D]:
        return cls._SINGLE_POLY_TYPE


class DopplerCentroid(_Poly2D):
    """DopplerCentroid class."""

    _UNITS = (
        "Hz",
        "Hz/s",
        "Hz/s",
        "Hz/s2",
        "Hz/s2",
        "Hz/s3",
        "Hz/s4",
        "Hz/s5",
        "Hz/s6",
        "Hz/s7",
        "Hz/s8",
    )


class DopplerCentroidVector(_Poly2DVector):
    """List of DopplerCentroid poly."""

    _SINGLE_POLY_TYPE = DopplerCentroid


class DopplerRate(_Poly2D):
    """DopplerRate class."""

    _UNITS = (
        "Hz/s",
        "Hz/s2",
        "Hz/s2",
        "Hz/s3",
        "Hz/s3",
        "Hz/s4",
        "Hz/s5",
        "Hz/s6",
        "Hz/s7",
        "Hz/s8",
        "Hz/s9",
    )


class DopplerRateVector(_Poly2DVector):
    """list of DopplerRate poly."""

    _SINGLE_POLY_TYPE = DopplerRate


class TopsAzimuthModulationRate(_Poly2D):
    """TopsAzimuthModulationRate class."""

    _UNITS = (
        "Hz",
        "Hz/s",
        "Hz/s",
        "Hz/s2",
        "Hz/s2",
        "Hz/s3",
        "Hz/s4",
        "Hz/s5",
        "Hz/s6",
        "Hz/s7",
        "Hz/s8",
    )


class TopsAzimuthModulationRateVector(_Poly2DVector):
    """List of TopsAzimuthModulationRate poly."""

    _SINGLE_POLY_TYPE = TopsAzimuthModulationRate


class SlantToGround(_Poly2D):
    """SlantToGround class."""

    _UNITS = (
        "m",
        "m/s",
        "m/s",
        "m/s2",
        "m/s2",
        "m/s3",
        "m/s4",
        "m/s5",
        "m/s6",
        "m/s7",
        "m/s8",
    )


class SlantToGroundVector(_Poly2DVector):
    """List of SlantToGround poly."""

    _SINGLE_POLY_TYPE = SlantToGround


class GroundToSlant(_Poly2D):
    """GroundToSlant class."""

    _UNITS = (
        "s",
        "s/m",
        "s/m",
        "s/m2",
        "s/m2",
        "s/m3",
        "s/m4",
        "s/m5",
        "s/m6",
        "s/m7",
        "s/m8",
    )


class GroundToSlantVector(_Poly2DVector):
    """List of GroundToSlant poly."""

    _SINGLE_POLY_TYPE = GroundToSlant


class SlantToIncidence(_Poly2D):
    """SlantToIncidence class."""

    _UNITS = (
        "Deg",
        "Deg/s",
        "Deg/s",
        "Deg/s2",
        "Deg/s2",
        "Deg/s3",
        "Deg/s4",
        "Deg/s5",
        "Deg/s6",
        "Deg/s7",
        "Deg/s8",
    )


class SlantToIncidenceVector(_Poly2DVector):
    """List of SlantToIncidence poly."""

    _SINGLE_POLY_TYPE = SlantToIncidence


class SlantToElevation(_Poly2D):
    """SlantToElevation class."""

    _UNITS = (
        "Deg",
        "Deg/s",
        "Deg/s",
        "Deg/s2",
        "Deg/s2",
        "Deg/s3",
        "Deg/s4",
        "Deg/s5",
        "Deg/s6",
        "Deg/s7",
        "Deg/s8",
    )


class SlantToElevationVector(_Poly2DVector):
    """List of SlantToElevation poly."""

    _SINGLE_POLY_TYPE = SlantToElevation


class AntennaInfo(MetaDataElement):
    """AntennaInfo class."""

    def __init__(
        self,
        i_sensor_name: str | None = None,
        i_polarization: str | EPolarization | None = None,
        i_acquisition_mode: str | None = None,
        i_acquisition_beam: str | None = None,
        i_lines_per_pattern: int = 0,
    ) -> None:
        self.sensor_name = i_sensor_name
        self.polarization = i_polarization
        self.acquisition_mode = i_acquisition_mode
        self.acquisition_beam = i_acquisition_beam
        self.lines_per_pattern = i_lines_per_pattern

    @property
    def polarization(self) -> EPolarization:
        return self._polarization

    @polarization.setter
    def polarization(self, i_polarization: str | EPolarization | None) -> None:
        self._polarization = EPolarization(i_polarization)


class DataStatistics(MetaDataElement):
    """DataStatistics class."""

    def __init__(
        self,
        i_num_samples: int = 0,
        i_max_i: float = 0.0,
        i_max_q: float = 0.0,
        i_min_i: float = 0.0,
        i_min_q: float = 0.0,
        i_sum_i: float = 0.0,
        i_sum_q: float = 0.0,
        i_sum_2_i: float = 0.0,
        i_sum_2_q: float = 0.0,
        i_std_dev_i: float = 0.0,
        i_std_dev_q: float = 0.0,
    ) -> None:
        self.num_samples = i_num_samples
        self.max_i = i_max_i
        self.max_q = i_max_q
        self.min_i = i_min_i
        self.min_q = i_min_q
        self.sum_i = i_sum_i
        self.sum_q = i_sum_q
        self.sum_2_i = i_sum_2_i
        self.sum_2_q = i_sum_2_q
        self.std_dev_i = i_std_dev_i
        self.std_dev_q = i_std_dev_q
        self._statistics_list: list[DataBlockStatistic] = []

    def add_data_block_statistic(self, i_data_block_statistic: DataBlockStatistic) -> None:
        if not isinstance(i_data_block_statistic, DataBlockStatistic):
            msg = "The input must be of type DataBlockStatistic"
            raise TypeError(msg)
        self._statistics_list.append(i_data_block_statistic)

    def get_data_block_statistic(self, index: int) -> DataBlockStatistic:
        if index < 0 or index >= self.get_number_of_data_block_statistic():
            msg = f"DataBlockStatistic not available for index: {index}"
            raise IndexError(msg)
        return self._statistics_list[index]

    def get_number_of_data_block_statistic(self) -> int:
        return len(self._statistics_list)


class DataBlockStatistic(MetaDataElement):
    """DataBlockStatistic class."""

    def __init__(
        self,
        i_line_start: int,
        i_line_stop: int,
        i_num_samples: int = 0,
        i_max_i: float = 0.0,
        i_max_q: float = 0.0,
        i_min_i: float = 0.0,
        i_min_q: float = 0.0,
        i_sum_i: float = 0.0,
        i_sum_q: float = 0.0,
        i_sum_2_i: float = 0.0,
        i_sum_2_q: float = 0.0,
    ) -> None:
        self.num_samples = i_num_samples
        self.max_i = i_max_i
        self.max_q = i_max_q
        self.min_i = i_min_i
        self.min_q = i_min_q
        self.sum_i = i_sum_i
        self.sum_q = i_sum_q
        self.sum_2_i = i_sum_2_i
        self.sum_2_q = i_sum_2_q
        self.line_start = i_line_start
        self.line_stop = i_line_stop


class CoregPoly(MetaDataElement):
    """CoregPoly class."""

    class _CoregPoly1DAz(_Poly2D):
        _UNITS = 10 * ("",)

    class _CoregPoly1DRg(_Poly2D):
        _UNITS = 10 * ("",)

    def __init__(
        self,
        i_ref_az: PreciseDateTime,
        i_ref_rg: float,
        i_coefficients_az: list[float],
        i_coefficients_rg: list[float],
    ) -> None:
        """Coregistration polynomials."""
        self._azimuth_coreg_poly = CoregPoly._CoregPoly1DAz(i_ref_az, i_ref_rg, i_coefficients_az)
        self._range_coreg_poly = CoregPoly._CoregPoly1DRg(i_ref_az, i_ref_rg, i_coefficients_rg)

    @property
    def ref_azimuth_time(self) -> PreciseDateTime:
        """Reference azimuth time."""
        return self._azimuth_coreg_poly.t_ref_az

    @ref_azimuth_time.setter
    def ref_azimuth_time(self, i_ref_az: PreciseDateTime) -> None:
        self._azimuth_coreg_poly.t_ref_az = i_ref_az
        self._range_coreg_poly.t_ref_az = i_ref_az

    @property
    def ref_range_time(self) -> float:
        """Reference range time."""
        return self._azimuth_coreg_poly.t_ref_rg

    @ref_range_time.setter
    def ref_range_time(self, i_ref_rg: float) -> None:
        self._azimuth_coreg_poly.t_ref_rg = i_ref_rg
        self._range_coreg_poly.t_ref_rg = i_ref_rg

    @property
    def azimuth_poly(self) -> _CoregPoly1DAz:
        """Azimuth coregistration polynomial."""
        return self._azimuth_coreg_poly

    @property
    def range_poly(self) -> _CoregPoly1DRg:
        """Range coregistration polynomial."""
        return self._range_coreg_poly


class CoregPolyVector(_Poly2DVector):
    """List of CoregPoly."""

    _SINGLE_POLY_TYPE = CoregPoly


class Pulse(MetaDataElement):
    """Pulse class."""

    def __init__(
        self,
        i_pulse_length: float | None = None,
        i_bandwidth: float | None = None,
        i_pulse_sampling_rate: float | None = None,
        i_pulse_energy: float | None = None,
        i_pulse_start_frequency: float | None = None,
        i_pulse_start_phase: float | None = None,
        i_pulse_direction: EPulseDirection | Literal["UP", "DOWN"] | None = None,
    ) -> None:
        """Pulse."""
        self.pulse_length = i_pulse_length
        self.pulse_length_unit = SECOND_STR
        self.bandwidth = i_bandwidth
        self.bandwidth_unit = HERTZ_STR
        self.pulse_energy = i_pulse_energy
        self.pulse_energy_unit = JOULE_STR
        self.pulse_sampling_rate = i_pulse_sampling_rate
        self.pulse_sampling_rate_unit = HERTZ_STR
        self.pulse_start_frequency = i_pulse_start_frequency
        self.pulse_start_frequency_unit = HERTZ_STR
        self.pulse_start_phase = i_pulse_start_phase
        self.pulse_start_phase_unit = RAD_STR
        self.pulse_direction = i_pulse_direction

    @property
    def pulse_direction(self) -> EPulseDirection | None:
        """Direction of pulse."""
        return self._pulse_direction

    @pulse_direction.setter
    def pulse_direction(self, i_pulse_direction: EPulseDirection | Literal["UP", "DOWN"] | None) -> None:
        if i_pulse_direction is not None:
            self._pulse_direction = EPulseDirection(i_pulse_direction)
        else:
            self._pulse_direction = i_pulse_direction


class MetaDataChannel:
    """MetaDataChannel class."""

    _supported_elements: ClassVar = [
        "RasterInfo",
        "DataSetInfo",
        "SwathInfo",
        "SamplingConstants",
        "AcquisitionTimeLine",
        "BurstInfo",
        "StateVectors",
        "AttitudeInfo",
        "Pulse",
        "GroundCornerPoints",
        "DopplerCentroidVector",
        "DopplerRateVector",
        "TopsAzimuthModulationRateVector",
        "SlantToGroundVector",
        "GroundToSlantVector",
        "SlantToIncidenceVector",
        "SlantToElevationVector",
        "AntennaInfo",
        "DataStatistics",
        "CoregPolyVector",
    ]

    def __init__(self) -> None:
        """MetaDataChannel."""
        self._contentID = None
        self._number = None
        self._total = None
        self._elements = collections.OrderedDict()
        for element_tag in self._supported_elements:
            self._elements[element_tag] = None
            setattr(self, element_tag, property(lambda s, element_tag=element_tag: s.get_element(element_tag)))

    @property
    def contentID(self) -> str | None:
        """MetaDataChannel content ID."""
        return self._contentID

    @contentID.setter
    def contentID(self, ID: str | None) -> None:
        self._contentID = ID

    @property
    def number(self) -> int | None:
        """Number of current MetadataChannel."""
        return self._number

    @number.setter
    def number(self, number: int | None) -> None:
        self._number = number

    @property
    def total(self) -> int | None:
        """Total number of MetaDataChannel."""
        return self._total

    @total.setter
    def total(self, total: int | None) -> None:
        self._total = total

    def __repr__(self) -> str:
        """Build MetaDataChannel string representation."""
        str_repr = ["\nMetaDataChannel\n\n"]
        if self.contentID is not None:
            str_repr += [f"ContentID={self.contentID}\n"]

        str_repr += [f"Number={self.number}\n"]
        str_repr += [f"Total={self.total}\n"]

        str_repr += [str(e) for e in self._elements.values()]
        return "".join(str_repr)

    def insert_element(self, element: MetaDataElement, *, overwrite_ok: bool = False) -> None:
        """Insert the specified metadata element.

        Parameters
        ----------
        element : MetaDataElement
            metadata element to insert
        overwrite_ok : bool
            overwrite existing metadata element of the same type

        """
        if self._elements[element.type()] is not None and not overwrite_ok:
            msg = (
                f"The element {element.type()} is already present in the current metadata channel: element not inserted"
            )
            warnings.warn(msg, stacklevel=2)
            return

        self._elements[element.type()] = element

    def remove_element(self, element_type: str) -> None:
        """Remove specified channel element.

        Parameters
        ----------
        element_type : str
            element name

        """
        self._elements[element_type] = None

    def get_element(self, element_type: str) -> MetaDataElement:
        """Get channel MetaDataElement from element name.

        Parameters
        ----------
        element_type : str
            element name

        Returns
        -------
        MetaDataElement
            channel MetaDataElement

        """
        return self._elements[element_type]

    @staticmethod
    def get_supported_metadata_elements() -> list[str]:
        """Retrieve the list of the supported channel elements.

        Returns
        -------
        List[str]
            list of elements name

        """
        return MetaDataChannel._supported_elements.copy()

    def get_sampling_constants(self) -> SamplingConstants:
        """SamplingConstants getter method.

        Returns
        -------
        SamplingConstants
            SamplingConstants MetaDataElement instance

        """
        return self.get_element("SamplingConstants")

    def get_pulse(self) -> Pulse:
        """Pulse getter method.

        Returns
        -------
        Pulse
            Pulse MetaDataElement instance

        """
        return self.get_element("Pulse")

    def get_raster_info(self) -> RasterInfo:
        """RasterInfo getter method.

        Returns
        -------
        RasterInfo
            RasterInfo MetaDataElement instance

        """
        return self.get_element("RasterInfo")

    def get_dataset_info(self) -> DataSetInfo:
        """DataSetInfo getter method.

        Returns
        -------
        DataSetInfo
            DataSetInfo MetaDataElement instance

        """
        return self.get_element("DataSetInfo")

    def get_state_vectors(self) -> StateVectors:
        """StateVectors getter method.

        Returns
        -------
        StateVectors
            StateVectors MetaDataElement instance

        """
        return self.get_element("StateVectors")

    def get_attitude_info(self) -> AttitudeInfo:
        """AttitudeInfo getter method.

        Returns
        -------
        AttitudeInfo
            AttitudeInfo MetaDataElement instance

        """
        return self.get_element("AttitudeInfo")

    def get_acquisition_time_line(self) -> AcquisitionTimeLine:
        """AcquisitionTimeLine getter method.

        Returns
        -------
        AcquisitionTimeLine
            AcquisitionTimeLine MetaDataElement instance

        """
        return self.get_element("AcquisitionTimeLine")

    def get_ground_corner_points(self) -> GroundCornerPoints:
        """GroundCornerPoints getter method.

        Returns
        -------
        GroundCornerPoints
            GroundCornerPoints MetaDataElement instance

        """
        return self.get_element("GroundCornerPoints")

    def get_burst_info(self) -> BurstInfo:
        """BurstInfo getter method.

        Returns
        -------
        BurstInfo
            BurstInfo MetaDataElement instance

        """
        return self.get_element("BurstInfo")

    def get_doppler_centroid(self) -> DopplerCentroidVector:
        """DopplerCentroidVector getter method.

        Returns
        -------
        DopplerCentroidVector
            DopplerCentroidVector MetaDataElement instance

        """
        return self.get_element("DopplerCentroidVector")

    def get_doppler_rate(self) -> DopplerRateVector:
        """DopplerRateVector getter method.

        Returns
        -------
        DopplerRateVector
            DopplerRateVector MetaDataElement instance

        """
        return self.get_element("DopplerRateVector")

    def get_tops_azimuth_modulation_rate(self) -> TopsAzimuthModulationRateVector:
        """TopsAzimuthModulationRateVector getter method.

        Returns
        -------
        TopsAzimuthModulationRateVector
            TopsAzimuthModulationRateVector MetaDataElement instance

        """
        return self.get_element("TopsAzimuthModulationRateVector")

    def get_slant_to_ground(self) -> SlantToGroundVector:
        """SlantToGroundVector getter method.

        Returns
        -------
        SlantToGroundVector
            SlantToGroundVector MetaDataElement instance

        """
        return self.get_element("SlantToGroundVector")

    def get_ground_to_slant(self) -> GroundToSlantVector:
        """GroundToSlantVector getter method.

        Returns
        -------
        GroundToSlantVector
            GroundToSlantVector MetaDataElement instance

        """
        return self.get_element("GroundToSlantVector")

    def get_slant_to_incidence(self) -> SlantToIncidence:
        """SlantToIncidence getter method.

        Returns
        -------
        SlantToIncidence
            SlantToIncidence MetaDataElement instance

        """
        return self.get_element("SlantToIncidence")

    def get_slant_to_elevation(self) -> SlantToElevation:
        """SlantToElevation getter method.

        Returns
        -------
        SlantToElevation
            SlantToElevation MetaDataElement instance

        """
        return self.get_element("SlantToElevation")

    def get_antenna_info(self) -> AntennaInfo:
        """AntennaInfo getter method.

        Returns
        -------
        AntennaInfo
            AntennaInfo MetaDataElement instance

        """
        return self.get_element("AntennaInfo")

    def get_data_statistics(self) -> DataStatistics:
        """DataStatistics getter method.

        Returns
        -------
        DataStatistics
            DataStatistics MetaDataElement instance

        """
        return self.get_element("DataStatistics")

    def get_swath_info(self) -> SwathInfo:
        """SwathInfo getter method.

        Returns
        -------
        SwathInfo
            SwathInfo MetaDataElement instance

        """
        return self.get_element("SwathInfo")

    def get_coreg_poly(self) -> CoregPolyVector:
        """CoregPolyVector getter method.

        Returns
        -------
        CoregPolyVector
            CoregPolyVector MetaDataElement instance

        """
        return self.get_element("CoregPolyVector")


class MetaData:
    """Metadata class.

    List of MetaDataChannels
    """

    def __init__(self, description: str = "") -> None:
        """Metadata."""
        self.description = description
        self._metadatachannels = []

    def append_channel(self, channel: MetaDataChannel) -> None:
        """Append the provided MetaDataChannel to the MetaData object.

        Parameters
        ----------
        channel : MetaDataChannel
            MetaDataChannel instance to be added

        """
        self._metadatachannels.append(channel)

    def insert_element(self, element: MetaDataElement, meta_data_ch_index: int = 0) -> None:
        """Insert a new metadata element into the selected metadata channel.

        Parameters
        ----------
        element : MetaDataElement
            metadata element to be inserted
        meta_data_ch_index : int, optional
            metadata channel number where to insert, by default 0

        """
        self.get_metadata_channels(channel_index=meta_data_ch_index).insert_element(element)

    def get_metadata_channels(self, channel_index: int = 0) -> MetaDataChannel:
        """Get the metadata channel instance corresponding to the specified channel index.

        Parameters
        ----------
        channel_index : int, optional
            channel of choice, by default 0

        Returns
        -------
        MetaDataChannel
            MetaDataChannel element corresponding to the specified channel

        """
        return self._metadatachannels[channel_index]

    def get_number_of_channels(self) -> int:
        """Get the umber of available channels in the metadata.

        Returns
        -------
        int
            number of channels

        """
        return len(self._metadatachannels)

    def get_sampling_constants(self, meta_data_ch_index: int = 0) -> SamplingConstants:
        """SamplingConstants getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        SamplingConstants
            SamplingConstants MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_sampling_constants()

    def get_pulse(self, meta_data_ch_index: int = 0) -> Pulse:
        """Pulse getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        Pulse
            Pulse MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_pulse()

    def get_raster_info(self, meta_data_ch_index: int = 0) -> RasterInfo:
        """RasterInfo getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        RasterInfo
            RasterInfo MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_raster_info()

    def get_dataset_info(self, meta_data_ch_index: int = 0) -> DataSetInfo:
        """DataSetInfo getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        DataSetInfo
            DataSetInfo MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_dataset_info()

    def get_state_vectors(self, meta_data_ch_index: int = 0) -> StateVectors:
        """StateVectors getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        StateVectors
            StateVectors MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_state_vectors()

    def get_attitude_info(self, meta_data_ch_index: int = 0) -> AttitudeInfo:
        """AttitudeInfo getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        AttitudeInfo
            AttitudeInfo MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_attitude_info()

    def get_acquisition_time_line(self, meta_data_ch_index: int = 0) -> AcquisitionTimeLine:
        """AcquisitionTimeLine getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        AcquisitionTimeLine
            AcquisitionTimeLine MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_acquisition_time_line()

    def get_ground_corner_points(self, meta_data_ch_index: int = 0) -> GroundCornerPoints:
        """GroundCornerPoints getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        GroundCornerPoints
            GroundCornerPoints MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_ground_corner_points()

    def get_burst_info(self, meta_data_ch_index: int = 0) -> BurstInfo:
        """BurstInfo getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        BurstInfo
            BurstInfo MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_burst_info()

    def get_doppler_centroid(self, meta_data_ch_index: int = 0) -> DopplerCentroidVector:
        """DopplerCentroidVector getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        DopplerCentroidVector
            DopplerCentroidVector MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_doppler_centroid()

    def get_doppler_rate(self, meta_data_ch_index: int = 0) -> DopplerRateVector:
        """DopplerRateVector getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        DopplerRateVector
            DopplerRateVector MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_doppler_rate()

    def get_tops_azimuth_modulation_rate(self, meta_data_ch_index: int = 0) -> TopsAzimuthModulationRateVector:
        """TopsAzimuthModulationRateVector getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        TopsAzimuthModulationRateVector
            TopsAzimuthModulationRateVector MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_tops_azimuth_modulation_rate()

    def get_slant_to_ground(self, meta_data_ch_index: int = 0) -> SlantToGroundVector:
        """SlantToGroundVector getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        SlantToGroundVector
            SlantToGroundVector MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_slant_to_ground()

    def get_ground_to_slant(self, meta_data_ch_index: int = 0) -> GroundToSlantVector:
        """GroundToSlantVector getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        GroundToSlantVector
            GroundToSlantVector MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_ground_to_slant()

    def get_slant_to_incidence(self, meta_data_ch_index: int = 0) -> SlantToIncidence:
        """SlantToIncidence getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        SlantToIncidence
            SlantToIncidence MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_slant_to_incidence()

    def get_slant_to_elevation(self, meta_data_ch_index: int = 0) -> SlantToElevation:
        """SlantToElevation getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        SlantToElevation
            SlantToElevation MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_slant_to_elevation()

    def get_antenna_info(self, meta_data_ch_index: int = 0) -> AntennaInfo:
        """AntennaInfo getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        AntennaInfo
            AntennaInfo MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_antenna_info()

    def get_data_statistics(self, meta_data_ch_index: int = 0) -> DataStatistics:
        """DataStatistics getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        DataStatistics
            DataStatistics MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_data_statistics()

    def get_swath_info(self, meta_data_ch_index: int = 0) -> SwathInfo:
        """SwathInfo getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        SwathInfo
            SwathInfo MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_swath_info()

    def get_coreg_poly(self, meta_data_ch_index: int = 0) -> CoregPolyVector:
        """CoregPolyVector getter method.

        Parameters
        ----------
        meta_data_ch_index : int, optional
            metadata channel of choice, by default 0

        Returns
        -------
        CoregPolyVector
            CoregPolyVector MetaDataElement instance

        """
        return self.get_metadata_channels(meta_data_ch_index).get_coreg_poly()
