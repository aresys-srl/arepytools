# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
MetaData module
-------------------------------
"""
import collections
import copy
import enum
import warnings
from abc import ABCMeta
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from ..timing.precisedatetime import PreciseDateTime

SECOND_STR = "s"
HERTZ_STR = "Hz"
JOULE_STR = "j"
RAD_STR = "rad"
UTC_STR = "Utc"


class EByteOrder(enum.Enum):
    """
    Byte orders supported ProductFolder's raster
    """

    be = "BIGENDIAN"
    le = "LITTLEENDIAN"


class ECellType(enum.Enum):
    """
    Data format supported in the ProductFolder' raster
    """

    int8 = "INT8"
    int16 = "INT16"
    int32 = "INT32"
    float32 = "FLOAT32"
    float64 = "FLOAT64"
    i8complex = "INT8_COMPLEX"
    i16complex = "INT16_COMPLEX"
    i32complex = "INT32_COMPLEX"
    fcomplex = "FLOAT_COMPLEX"
    dcomplex = "DOUBLE_COMPLEX"
    custom = "CUSTOM"


class EOrbitDirection(enum.Enum):
    """
    Satellite orbit direction
    """

    ascending = "ASCENDING"
    descending = "DESCENDING"


class ESideLooking(enum.Enum):
    """
    Satellite side looking
    """

    right_looking = "RIGHT"
    left_looking = "LEFT"


class EPolarization(enum.Enum):
    """
    Polarizations
    """

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
    """
    Chirp pulse type
    """

    up = "UP"
    down = "DOWN"


class EReferenceFrame(enum.Enum):
    """
    Data reference frame
    """

    geocentric = "GEOCENTRIC"
    geodetic = "GEODETIC"
    zerodoppler = "ZERODOPPLER"
    none = ""


class ERotationOrder(enum.Enum):
    """
    Attitude rotation order
    - y: yaw
    - p: pitch
    - r: roll
    """

    ypr = "ypr"
    yrp = "yrp"
    pry = "pry"
    pyr = "pyr"
    ryp = "ryp"
    rpy = "rpy"
    none = ""


class EAttitudeType(enum.Enum):
    """
    Attitude type
    """

    nominal = "NOMINAL"
    refined = "REFINED"
    none = ""


class ERasterFormatType(enum.Enum):
    """
    Aresys raster format
    """

    aresys_raster = "ARESYS_RASTER"
    aresys_geotiff = "ARESYS_GEOTIFF"
    raster = "ARESYS_RASTER"


class MetaDataElement(metaclass=ABCMeta):
    """
    Base class for metadata elements
    """

    TYPE = None

    def __repr__(self):
        max_len = max([len(x) for x in self.__dict__])
        str_repr = ["\nMetaDataElement: {}\n\n".format(self.TYPE)]
        str_repr += [
            "{elemName:>{length}}: {value}\n".format(
                elemName=k.lstrip("_.- "), length=max_len + 1, value=v
            )
            for k, v in self.__dict__.items()
        ]
        return "".join(str_repr)

    def type(self):
        return self.TYPE

    def copy(self):
        return copy.copy(self)


class RasterInfo(MetaDataElement):
    """
    RasterInfo class
    """

    TYPE = "RasterInfo"

    def __init__(
        self,
        lines,
        samples,
        celltype: Union[str, ECellType],
        filename=None,
        header_offset_bytes=0,
        row_prefix_bytes=0,
        byteorder: Union[str, EByteOrder] = "LITTLEENDIAN",
        invalid_value=None,
        format_type=None,
    ):
        self._file_name = filename

        if isinstance(lines, int):
            self._lines = lines
        else:
            raise ValueError("The lines parameter have to be an integer")

        if isinstance(samples, int):
            self._samples = samples
        else:
            raise ValueError("The samples parameter have to be an integer")

        if isinstance(header_offset_bytes, int):
            self._header_offset_bytes = header_offset_bytes
        else:
            raise ValueError("The header offset parameter have to be an integer")

        if isinstance(row_prefix_bytes, int):
            self._row_prefix_bytes = row_prefix_bytes
        else:
            raise ValueError("The row prefix parameter have to be an integer")

        self._byte_order = EByteOrder(byteorder)
        self._cell_type = ECellType(celltype)

        self._lines_start = 0.0
        self._lines_start_unit = ""
        self._lines_step = 0.0
        self._lines_step_unit = ""

        self._samples_start = 0.0
        self._samples_start_unit = ""
        self._samples_step = 0.0
        self._samples_step_unit = ""

        self._invalid_value = invalid_value

        if format_type is not None:
            format_type = ERasterFormatType(format_type)
        self._format_type = format_type

    @property
    def file_name(self):
        return self._file_name

    @property
    def lines(self):
        return self._lines

    @property
    def samples(self):
        return self._samples

    @property
    def header_offset_bytes(self):
        return self._header_offset_bytes

    @property
    def row_prefix_bytes(self):
        return self._row_prefix_bytes

    @property
    def lines_start(self):
        return self._lines_start

    @property
    def lines_start_unit(self):
        return self._lines_start_unit

    @property
    def lines_step(self):
        return self._lines_step

    @property
    def lines_step_unit(self):
        return self._lines_step_unit

    @property
    def samples_start(self):
        return self._samples_start

    @property
    def samples_start_unit(self):
        return self._samples_start_unit

    @property
    def samples_step(self):
        return self._samples_step

    @property
    def samples_step_unit(self):
        return self._samples_step_unit

    @property
    def byte_order(self):
        return self._byte_order

    @property
    def cell_type(self):
        return self._cell_type

    @property
    def invalid_value(self):
        return self._invalid_value

    @property
    def format_type(self):
        return self._format_type

    def set_lines_axis(
        self,
        lines_start,
        lines_start_unit: str,
        lines_step: float,
        lines_step_unit: str,
    ):
        """
        setter of the RasterInfo lines axis
        :param lines_start:
        :param lines_start_unit:
        :param lines_step:
        :param lines_step_unit:
        """
        self._lines_start = lines_start
        self._lines_start_unit = lines_start_unit
        self._lines_step = lines_step
        self._lines_step_unit = lines_step_unit

    def set_samples_axis(
        self,
        samples_start,
        samples_start_unit: str,
        samples_step: float,
        samples_step_unit: str,
    ):
        self._samples_start = samples_start
        self._samples_start_unit = samples_start_unit
        self._samples_step = samples_step
        self._samples_step_unit = samples_step_unit


class DataSetInfo(MetaDataElement):
    """
    DataSetInfo class
    """

    TYPE = "DataSetInfo"

    def __init__(self, acquisition_mode_i=None, fc_hz_i=None):
        self.sensor_name: Optional[str] = None
        self.description: Optional[str] = None
        self._sense_date = None
        self.acquisition_mode: Optional[str] = acquisition_mode_i
        self.image_type: Optional[str] = None
        self.projection: Optional[str] = None
        self.acquisition_station: Optional[str] = None
        self.processing_center: Optional[str] = None
        self._processing_date = None
        self.processing_software: Optional[str] = None
        self.fc_hz = fc_hz_i
        self._side_looking = None
        self.external_calibration_factor: Optional[float] = None
        self.data_take_id: Optional[int] = None

    @property
    def sense_date(self):
        return self._sense_date

    @sense_date.setter
    def sense_date(self, sense_date_i: Optional[PreciseDateTime]):
        if isinstance(sense_date_i, PreciseDateTime) or sense_date_i is None:
            self._sense_date = sense_date_i
        else:
            raise ValueError("Sense date have to be a PreciseDateTime or 'None'")

    @property
    def processing_date(self):
        return self._processing_date

    @processing_date.setter
    def processing_date(self, processing_date_i: Optional[PreciseDateTime]):
        if isinstance(processing_date_i, PreciseDateTime) or processing_date_i is None:
            self._processing_date = processing_date_i
        else:
            raise ValueError("Processing date have to be a PreciseDateTime or 'None'")

    @property
    def fc_hz(self):
        return self._fc_hz

    @fc_hz.setter
    def fc_hz(self, fc_hz_i):
        self._fc_hz = fc_hz_i

    @property
    def side_looking(self):
        return self._side_looking

    @side_looking.setter
    def side_looking(self, side_looking_i):
        self._side_looking = ESideLooking(side_looking_i)


class GeoPoint(MetaDataElement):
    """
    GeoPoint class
    """

    TYPE = "GeoPoint"

    def __init__(self, lat=0.0, lon=0.0, height=0.0, theta_inc=0.0, theta_look=0.0):
        self.lat = lat
        self.lon = lon
        self.height = height
        self.theta_inc = theta_inc
        self.theta_look = theta_look

    def to_list(self):
        return [self.lat, self.lon, self.height, self.theta_inc, self.theta_look]


class GroundCornerPoints(MetaDataElement):
    """
    GrondCornerPoint class
    """

    TYPE = "GroundCornerPoints"

    def __init__(self):
        self.easting_grid_size = 0.0
        self.northing_grid_size = 0.0
        self.geo_points = [GeoPoint() for _ in range(5)]

    @property
    def center_point(self):
        return self.geo_points[4]

    @center_point.setter
    def center_point(self, center: GeoPoint):
        self.geo_points[4] = center

    @property
    def ne_point(self):
        return self.geo_points[1]

    @ne_point.setter
    def ne_point(self, ne_point: GeoPoint):
        self.geo_points[1] = ne_point

    @property
    def nw_point(self):
        return self.geo_points[0]

    @nw_point.setter
    def nw_point(self, nw_point: GeoPoint):
        self.geo_points[0] = nw_point

    @property
    def se_point(self):
        return self.geo_points[3]

    @se_point.setter
    def se_point(self, se_point: GeoPoint):
        self.geo_points[3] = se_point

    @property
    def sw_point(self):
        return self.geo_points[2]

    @sw_point.setter
    def sw_point(self, sw_point: GeoPoint):
        self.geo_points[2] = sw_point


class SwathInfo(MetaDataElement):
    """
    SwathInfo class
    """

    TYPE = "SwathInfo"

    def __init__(self, swath_i=None, polarization_i=None, acquisition_prf_i=0.0):
        self.swath: Optional[str] = swath_i
        self.polarization = polarization_i
        self.acquisition_prf = acquisition_prf_i
        self.acquisition_prf_unit = HERTZ_STR
        self.swath_acquisition_order = 0
        self.rank = 0
        self.range_delay_bias = 0.0
        self.range_delay_bias_unit = SECOND_STR
        self.acquisition_start_time = None
        self.acquisition_start_time_unit = UTC_STR
        self.azimuth_steering_rate_reference_time = 0.0
        self.az_steering_rate_ref_time_unit = SECOND_STR
        self.echoes_per_burst = 0
        self.azimuth_steering_rate_pol = (0, 0, 0)
        self.rx_gain: Optional[float] = None
        self.channel_delay: Optional[float] = None

    @property
    def polarization(self):
        return self._polarization

    @polarization.setter
    def polarization(self, polarization_i):
        self._polarization = EPolarization(polarization_i)

    @property
    def acquisition_prf(self):
        return self._acquisition_prf

    @acquisition_prf.setter
    def acquisition_prf(self, acquisition_prf_i):
        self._acquisition_prf = acquisition_prf_i

    @property
    def acquisition_start_time(self):
        return self._acquisition_start_time

    @acquisition_start_time.setter
    def acquisition_start_time(self, acquisition_start_time_i):
        if isinstance(acquisition_start_time_i, PreciseDateTime):
            self._acquisition_start_time = acquisition_start_time_i
        elif acquisition_start_time_i is None:
            self._acquisition_start_time = None
        else:
            raise TypeError("Acquisition start time has to be a PreciseDateTime")

    @property
    def azimuth_steering_rate_pol(self):
        return self._azimuth_steering_rate_pol

    @azimuth_steering_rate_pol.setter
    def azimuth_steering_rate_pol(self, i_azimuth_steering_pol):
        if len(i_azimuth_steering_pol) == 3 and isinstance(
            i_azimuth_steering_pol, tuple
        ):
            self._azimuth_steering_rate_pol = i_azimuth_steering_pol
        else:
            raise TypeError(
                "The azimuth steering rate pol has to be a tuple of 3 elements"
            )


class SamplingConstants(MetaDataElement):
    """
    SamplingConstants class
    """

    TYPE = "SamplingConstants"

    def __init__(self, frg_hz_i=None, brg_hz_i=None, faz_hz_i=None, baz_hz_i=None):
        self.frg_hz = frg_hz_i
        self.frg_hz_unit = HERTZ_STR
        self.brg_hz = brg_hz_i
        self.brg_hz_unit = HERTZ_STR
        self.faz_hz = faz_hz_i
        self.faz_hz_unit = HERTZ_STR
        self.baz_hz = baz_hz_i
        self.baz_hz_unit = HERTZ_STR


class AcquisitionTimeLine(MetaDataElement):
    """
    AcquisitionTimeLine class
    """

    TYPE = "AcquisitionTimeLine"

    def __init__(
        self,
        missing_lines_number_i=0,
        missing_lines_azimuth_times_i=None,
        swst_changes_number_i=0,
        swst_changes_azimuth_times_i=None,
        swst_changes_values_i=None,
        noise_packets_number_i=0,
        noise_packets_azimuth_times_i=None,
        internal_calibration_number_i=0,
        internal_calibration_azimuth_times_i=None,
        swl_changes_number_i=0,
        swl_changes_azimuth_times_i=None,
        swl_changes_values_i=None,
        prf_changes_number_i=0,
        prf_changes_azimuth_times_i=None,
        prf_changes_values_i=None,
        chirp_period: Optional[str] = None,
    ):
        integer_arguments = (
            missing_lines_number_i,
            swst_changes_number_i,
            noise_packets_number_i,
            internal_calibration_number_i,
            swl_changes_number_i,
            prf_changes_number_i,
        )

        integer_tags = (
            "missing_lines_number_i",
            "swst_changes_number_i",
            "noise_packets_number_i",
            "internal_calibration_number_i",
            "swl_changes_number_i",
            "prf_changes_number_i",
        )

        size_to_vec = dict()
        size_to_vec["missing_lines_number_i"] = (
            missing_lines_number_i,
            [missing_lines_azimuth_times_i],
        )
        size_to_vec["swst_changes_number_i"] = (
            swst_changes_number_i,
            [swst_changes_azimuth_times_i, swst_changes_values_i],
        )
        size_to_vec["noise_packets_number_i"] = (
            noise_packets_number_i,
            [noise_packets_azimuth_times_i],
        )
        size_to_vec["internal_calibration_number_i"] = (
            internal_calibration_number_i,
            [internal_calibration_azimuth_times_i],
        )
        size_to_vec["swl_changes_number_i"] = (
            swl_changes_number_i,
            [swl_changes_azimuth_times_i, swl_changes_values_i],
        )
        size_to_vec["prf_changes_number_i"] = (
            prf_changes_number_i,
            [prf_changes_azimuth_times_i, prf_changes_values_i],
        )

        for arg, tag in zip(integer_arguments, integer_tags):
            if not isinstance(arg, int):
                if isinstance(arg, float):
                    if abs(int(arg) - arg) == 0.0:
                        arg = int(arg)
                    else:
                        raise ValueError(
                            "{} should be an integer not a float".format(tag)
                        )
                else:
                    try:
                        arg = int(arg)
                    except TypeError as exc:
                        raise ValueError(
                            "{} wrong type: {} != int".format(tag, type(arg))
                        ) from exc
            if arg < 0:
                raise ValueError("{} should be non-negative".format(tag))

        for tag, tup in size_to_vec.items():
            for vec in tup[1]:
                if vec is not None:
                    if len(vec) != tup[0]:
                        raise ValueError(
                            "Incorrect size of vectors ({}) {} != {}".format(
                                tag, len(vec), tup[0]
                            )
                        )

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
    def duplicated_lines(self):
        return self._duplicated_lines_number, self._duplicated_lines_azimuth_times

    @duplicated_lines.setter
    def duplicated_lines(self, duplicated_lines):
        if len(duplicated_lines[1]) != duplicated_lines[0]:
            raise ValueError(
                "Duplicated lines inconsistent tuple: {} != {}".format(
                    duplicated_lines[0], len(duplicated_lines[1])
                )
            )

        self._duplicated_lines_number = duplicated_lines[0]
        self._duplicated_lines_azimuth_times = duplicated_lines[1]

    @property
    def internal_calibration(self):
        return (
            self._internal_calibration_number,
            self._internal_calibration_azimuth_times,
        )

    @internal_calibration.setter
    def internal_calibration(self, internal_calibration):
        self._internal_calibration_number = len(internal_calibration)
        self._internal_calibration_azimuth_times = internal_calibration

    @property
    def missing_lines(self) -> Tuple[int, Optional[List[float]]]:
        return self._missing_lines_number, self._missing_lines_azimuth_times

    @missing_lines.setter
    def missing_lines(self, azimuth_times):
        self._missing_lines_number = len(azimuth_times)
        self._missing_lines_azimuth_times = azimuth_times

    @property
    def noise_packet(self):
        return self._noise_packets_number, self._noise_packets_azimuth_times

    @noise_packet.setter
    def noise_packet(self, azimuth_times):
        self._noise_packets_number = len(azimuth_times)
        self._noise_packets_azimuth_times = azimuth_times

    @property
    def swl_changes(self):
        return (
            self._swl_changes_number,
            self._swl_changes_azimuth_times,
            self._swl_changes_values,
        )

    @swl_changes.setter
    def swl_changes(self, swl_changes):
        if swl_changes[0] != len(swl_changes[1]):
            raise ValueError("Inconsistent swl changes sizes")
        self._swl_changes_number = swl_changes[0]
        self._swl_changes_azimuth_times = swl_changes[1]
        self._swl_changes_values = swl_changes[2]

    @property
    def swst_changes(self) -> Tuple[int, Optional[List[float]], Optional[List[float]]]:
        return (
            self._swst_changes_number,
            self._swst_changes_azimuth_times,
            self._swst_changes_values,
        )

    @swst_changes.setter
    def swst_changes(self, swst_changes):
        if swst_changes[0] != len(swst_changes[1]):
            raise ValueError("Inconsistent swst changes sizes")
        self._swst_changes_number = swst_changes[0]
        self._swst_changes_azimuth_times = swst_changes[1]
        self._swst_changes_values = swst_changes[2]

    @property
    def prf_changes(self):
        return (
            self._prf_changes_number,
            self._prf_changes_azimuth_times,
            self._prf_changes_values,
        )

    @prf_changes.setter
    def prf_changes(self, prf_changes):
        if prf_changes[0] != len(prf_changes[1]):
            raise ValueError("Inconsistent prf changes sizes")
        self._prf_changes_number = prf_changes[0]
        self._prf_changes_azimuth_times = prf_changes[1]
        self._prf_changes_values = prf_changes[2]


class AttitudeInfo(MetaDataElement):
    """
    AttitudeInfo class
    """

    TYPE = "AttitudeInfo"

    _default_attitude_type = EAttitudeType("NOMINAL")

    def __init__(
        self,
        yaw: Optional[npt.ArrayLike] = None,
        pitch: Optional[npt.ArrayLike] = None,
        roll: Optional[npt.ArrayLike] = None,
        t0=None,
        delta_t=0.0,
        ref_frame: Optional[str] = None,
        rot_order: Optional[str] = None,
    ):
        if ref_frame is None:
            ref_frame = ""
        if rot_order is None:
            rot_order = ""
        if t0 is None:
            t0 = PreciseDateTime()

        self._reset_angles(yaw, pitch, roll)

        self._reference_frame = None
        self.reference_frame = ref_frame
        self._rotation_order = None
        self.rotation_order = rot_order

        self._attitude_type = self._default_attitude_type

        self._t_ref_Utc = None
        self.reference_time = t0

        self._dtYPR_s = None
        self.time_step = delta_t

    def _reset_angles(
        self,
        yaw: Optional[npt.ArrayLike],
        pitch: Optional[npt.ArrayLike],
        roll: Optional[npt.ArrayLike],
    ):
        if yaw is None and pitch is None and roll is None:
            return
        yaw_vector = np.array(yaw)
        pitch_vector = np.array(pitch)
        roll_vector = np.array(roll)

        yaw_size = yaw_vector.size
        for vec, tag in zip(
            (yaw_vector, pitch_vector, roll_vector), ("yaw", "pitch", "roll")
        ):
            wrong_dimension_string = (
                "Provided {} vector shall be a 1xN or Nx1 array".format(tag)
            )
            if vec.ndim == 1:
                if vec.size != yaw_size:
                    raise ValueError(wrong_dimension_string)
            elif vec.ndim == 2:
                if vec.shape[0] != 1 and vec.shape[1] != 1:
                    raise ValueError(wrong_dimension_string)
            else:
                raise ValueError(wrong_dimension_string)

            if yaw_size != vec.size:
                raise ValueError(
                    "{} and yaw vectors must have the same number of elements".format(
                        tag
                    )
                )
        self._yaw_deg = yaw_vector
        self._pitch_deg = pitch_vector
        self._roll_deg = roll_vector
        self._nYPR_n = yaw_size

    @property
    def reference_frame(self) -> EReferenceFrame:
        return self._reference_frame

    @reference_frame.setter
    def reference_frame(self, reference_frame: str):
        self._reference_frame = EReferenceFrame(reference_frame.upper())

    @property
    def rotation_order(self) -> ERotationOrder:
        return self._rotation_order

    @rotation_order.setter
    def rotation_order(self, rotation_order: str):
        self._rotation_order = ERotationOrder(rotation_order.lower())

    @property
    def attitude_records_number(self):
        return self._nYPR_n

    @property
    def attitude_type(self) -> EAttitudeType:
        return self._attitude_type

    @attitude_type.setter
    def attitude_type(self, attitude_type: str):
        self._attitude_type = EAttitudeType(attitude_type.upper())

    @property
    def pitch_vector(self):
        return self._pitch_deg

    @property
    def reference_time(self):
        return self._t_ref_Utc

    @reference_time.setter
    def reference_time(self, reference_time):
        if not isinstance(reference_time, (int, float, PreciseDateTime)):
            raise ValueError("Input start time must be a scalar PreciseDateTime")
        self._t_ref_Utc = reference_time

    @property
    def roll_vector(self):
        return self._roll_deg

    @property
    def time_step(self):
        return self._dtYPR_s

    @time_step.setter
    def time_step(self, delta_t):
        self._dtYPR_s = delta_t

    @property
    def yaw_vector(self):
        return self._yaw_deg

    def set_attitude_angles_vectors(self, yaw, pitch, roll):
        self._reset_angles(yaw, pitch, roll)


class Burst(MetaDataElement):
    """
    Single Burst class
    """

    TYPE = "Burst"

    def __init__(
        self,
        range_start_time_i,
        azimuth_start_time_i,
        lines_i,
        burst_center_azimuth_shift_i=None,
    ):
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
    def burst_center_azimuth_shift(self) -> Optional[float]:
        return self._burst_center_azimuth_shift

    @property
    def lines(self) -> int:
        return self._lines


class BurstInfo(MetaDataElement):
    """
    BurstInfo class
    """

    TYPE = "BurstInfo"

    def __init__(self, burst_repetition_frequency=0.0):
        self._bursts = list()
        self._lines_per_burst_present = False
        self._lines_per_burst = 0
        self.burst_repetition_frequency = burst_repetition_frequency

    def is_lines_per_burst_present(self):
        return self._lines_per_burst_present

    @property
    def lines_per_burst(self):
        return self._lines_per_burst

    def add_burst(
        self,
        range_start_time_i,
        azimuth_start_time_i,
        lines_i,
        burst_center_azimuth_shift_i=None,
    ):
        if len(self._bursts) == 0:
            self._lines_per_burst_present = True
            self._lines_per_burst = lines_i
        else:
            if self._lines_per_burst_present:
                if self._lines_per_burst != lines_i:
                    self._lines_per_burst_present = False

        if burst_center_azimuth_shift_i is None:
            burst_ = Burst(range_start_time_i, azimuth_start_time_i, lines_i)
        else:
            burst_ = Burst(
                range_start_time_i,
                azimuth_start_time_i,
                lines_i,
                burst_center_azimuth_shift_i,
            )
        self._bursts.append(burst_)

    def get_number_of_bursts(self):
        return len(self._bursts)

    def get_burst(self, burst_index) -> Burst:
        if 0 <= burst_index < self.get_number_of_bursts():
            return self._bursts[burst_index]

        raise RuntimeError("Burst index out of range")

    def clear_bursts(self):
        self._bursts = list()

    def _get_bursts_property(self, burst_index, attr):
        if burst_index is None:
            return np.asarray([getattr(burst, attr) for burst in self._bursts])

        if burst_index < 0 or burst_index >= self.get_number_of_bursts():
            raise ValueError("Not valid butst index")

        return getattr(self._bursts[burst_index], attr)

    def get_lines(self, burst_index=None):
        return self._get_bursts_property(burst_index, "lines")

    def get_azimuth_start_time(self, burst_index=None):
        return self._get_bursts_property(burst_index, "azimuth_start_time")

    def get_burst_center_azimuth_shift(self, burst_index=None):
        return self._get_bursts_property(burst_index, "burst_center_azimuth_shift")

    def get_range_start_time(self, burst_index=None):
        return self._get_bursts_property(burst_index, "range_start_time")

    def get_burst_roi(self, burst_index, raster_info: RasterInfo, roi_range=None):
        if roi_range is None:
            roi_range = [0, raster_info.samples]

        accumulated_lines = 0
        for i_burst in range(0, burst_index):
            accumulated_lines += self.get_lines(i_burst)

        first_line = accumulated_lines
        return [first_line, roi_range[0], self.get_lines(burst_index), roi_range[1]]


class StateVectors(MetaDataElement):
    """
    StateVectors class
    """

    TYPE = "StateVectors"

    def __init__(
        self,
        position_vector: Optional[np.ndarray] = None,
        velocity_vector: Optional[np.ndarray] = None,
        t_ref_utc=None,
        dt_sv_s=0.0,
    ):
        position_vector = np.array(position_vector)
        velocity_vector = np.array(velocity_vector)

        if position_vector.ndim > 2 or position_vector.size % 3 != 0:
            raise ValueError("Wrong array size for input position vector")
        if velocity_vector.ndim > 2 or velocity_vector.size % 3 != 0:
            raise ValueError("Wrong array size for input velocity vector")

        self._position_vector = position_vector
        self._velocity_vector = velocity_vector
        self._t_ref_utc = t_ref_utc
        self._dt_sv_s = dt_sv_s
        self._orbit_number = -1
        self._track_number = -1
        self._anx_time = None
        self._anx_position = None

    @property
    def anx_position(self):
        return self._anx_position

    def get_anx_time(self):
        """
        .. deprecated:: v1.1.0
            Use :data:`anx_time` property instead.
        """
        warnings.warn(
            "get_anx_time is deprecated: use anx_time instead",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.anx_time

    @property
    def anx_time(self):
        return self._anx_time

    @property
    def orbit_direction(self):
        if self._velocity_vector[0][2] > 0:
            return EOrbitDirection.ascending

        return EOrbitDirection.descending

    @property
    def orbit_number(self):
        return self._orbit_number

    @property
    def position_vector(self):
        return self._position_vector

    @property
    def reference_time(self):
        return self._t_ref_utc

    @property
    def number_of_state_vectors(self) -> int:
        return int(self._position_vector.size / 3)

    @property
    def time_step(self):
        return self._dt_sv_s

    @property
    def track_number(self):
        return self._track_number

    @property
    def velocity_vector(self):
        return self._velocity_vector

    def set_axn_info(self, i_anx_time, i_anx_pos):
        """
        .. deprecated:: v1.1.0
            use :func:`set_anx_info` instead.
        """
        warnings.warn(
            "set_axn_info is deprecated: use set_anx_info instead",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.set_anx_info(i_anx_time, i_anx_pos)

    def set_anx_info(self, i_anx_time, i_anx_pos):
        assert (self._anx_time is not None and self._anx_position is not None) or (
            self._anx_time is None and self._anx_position is None
        )

        if (i_anx_time is None and i_anx_pos is not None) or (
            i_anx_time is not None and i_anx_pos is None
        ):
            if i_anx_time is not None:
                raise ValueError("It is not allowed to specify ANX time only")

            assert i_anx_pos is not None
            raise ValueError("It is not allowed to specify ANX position only")

        assert (i_anx_time is not None and i_anx_pos is not None) or (
            i_anx_time is None and i_anx_pos is None
        )

        if i_anx_pos is not None and (
            not isinstance(i_anx_pos, list) or len(i_anx_pos) != 3
        ):
            raise TypeError(
                "Wrong input type for ANX position. It must be None or a list of 3 elements"
            )

        if i_anx_time is not None and not isinstance(i_anx_time, PreciseDateTime):
            raise TypeError(
                "Wrong input type for ANX time. It must be None or a PreciseDateTime object"
            )

        self._anx_position = i_anx_pos
        self._anx_time = i_anx_time

    @orbit_number.setter
    def orbit_number(self, i_orbit_number):
        if i_orbit_number <= 0 or not isinstance(i_orbit_number, int):
            raise ValueError(
                "Provided orbit number must have an integer and positive value."
            )
        self._orbit_number = i_orbit_number

    def set_state_vectors(self, i_position, i_velocity, i_reference_time, i_time_step):
        raise NotImplementedError("")

    @track_number.setter
    def track_number(self, i_track_number):
        if not isinstance(i_track_number, int) or i_track_number <= 0:
            raise ValueError(
                "Provided track number must have an integer and positive value."
            )
        self._track_number = i_track_number


class _Poly2D(MetaDataElement):
    """
    Base class for Poly2D format
    """

    TYPE = "Poly2D"
    _POWERS_X = (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0)

    _POWERS_Y = (0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8)

    _UNITS = 10 * ("",)

    def __init__(self, i_ref_az=None, i_ref_rg=None, i_coefficients=None):
        self.t_ref_az = i_ref_az
        self.t_ref_rg = i_ref_rg
        if len(i_coefficients) > len(self._POWERS_X):
            raise ValueError("the size of coefficients and powers must agree")
        self._coefficients = i_coefficients

    @property
    def coefficients(self):
        return self._coefficients

    @staticmethod
    def get_powers_x():
        return _Poly2D._POWERS_X

    @staticmethod
    def get_powers_y():
        return _Poly2D._POWERS_Y

    @classmethod
    def get_units(cls):
        return cls._UNITS


class _Poly2DVector(MetaDataElement):
    """
    Base class for list of Poly2D
    """

    TYPE = "Poly2DVector"
    _SINGLE_POLY_TYPE = _Poly2D

    def __init__(self, i_poly2d: Optional[list] = None):
        self._poly_list = i_poly2d if i_poly2d is not None else list()
        assert isinstance(self._poly_list, list), "The input should be a list"
        self._current_poly_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_poly_index >= self.get_number_of_poly():
            self._current_poly_index = 0
            raise StopIteration
        self._current_poly_index += 1
        return self.get_poly(self._current_poly_index - 1)

    def __len__(self):
        return len(self._poly_list)

    def add_poly(self, i_poly2d):
        self._poly_list.append(i_poly2d)

    def get_number_of_poly(self):
        return len(self._poly_list)

    def get_poly(self, index):
        if 0 <= index < self.get_number_of_poly():
            return self._poly_list[index]

        raise IndexError("Polynomial not available for index: {}".format(index))

    @classmethod
    def get_single_poly_type(cls):
        return cls._SINGLE_POLY_TYPE


class DopplerCentroid(_Poly2D):
    """
    DopplerCentroid class
    """

    TYPE = "DopplerCentroid"

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
    """
    List of DopplerCentroid poly
    """

    TYPE = "DopplerCentroidVector"
    _SINGLE_POLY_TYPE = DopplerCentroid


class DopplerRate(_Poly2D):
    """
    DopplerRate class
    """

    TYPE = "DopplerRate"

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
    """
    list of DopplerRate poly
    """

    TYPE = "DopplerRateVector"
    _SINGLE_POLY_TYPE = DopplerRate


class TopsAzimuthModulationRate(_Poly2D):
    """
    TopsAzimuthModulationRate class
    """

    TYPE = "TopsAzimuthModulationRate"

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
    """
    List of TopsAzimuthModulationRate poly
    """

    TYPE = "TopsAzimuthModulationRateVector"
    _SINGLE_POLY_TYPE = TopsAzimuthModulationRate


class SlantToGround(_Poly2D):
    """
    SlantToGround class
    """

    TYPE = "SlantToGround"

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
    """
    List of SlantToGround poly
    """

    TYPE = "SlantToGroundVector"
    _SINGLE_POLY_TYPE = SlantToGround


class GroundToSlant(_Poly2D):
    """
    GroundToSlant class
    """

    TYPE = "GroundToSlant"

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
    """
    List of GroundToSlant poly
    """

    TYPE = "GroundToSlantVector"
    _SINGLE_POLY_TYPE = GroundToSlant


class SlantToIncidence(_Poly2D):
    """
    SlantToIncidence class
    """

    TYPE = "SlantToIncidence"

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
    """
    List of SlantToIncidence poly
    """

    TYPE = "SlantToIncidenceVector"
    _SINGLE_POLY_TYPE = SlantToIncidence


class SlantToElevation(_Poly2D):
    """
    SlantToElevation class
    """

    TYPE = "SlantToElevation"

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
    """
    List of SlantToElevation poly
    """

    TYPE = "SlantToElevationVector"
    _SINGLE_POLY_TYPE = SlantToElevation


class AntennaInfo(MetaDataElement):
    """
    AntennaInfo class
    """

    TYPE = "AntennaInfo"

    def __init__(
        self,
        i_sensor_name=None,
        i_polarization=None,
        i_acquisition_mode=None,
        i_acquisition_beam=None,
        i_lines_per_pattern=0,
    ):
        self.sensor_name = i_sensor_name
        self.polarization = i_polarization
        self.acquisition_mode = i_acquisition_mode
        self.acquisition_beam = i_acquisition_beam
        self.lines_per_pattern = i_lines_per_pattern

    @property
    def polarization(self):
        return self._polarization

    @polarization.setter
    def polarization(self, i_polarization):
        self._polarization = EPolarization(i_polarization)


class DataStatistics(MetaDataElement):
    """
    DataStatistics class
    """

    TYPE = "DataStatistics"

    def __init__(
        self,
        i_num_samples=0,
        i_max_i=0.0,
        i_max_q=0.0,
        i_min_i=0.0,
        i_min_q=0.0,
        i_sum_i=0.0,
        i_sum_q=0.0,
        i_sum_2_i=0.0,
        i_sum_2_q=0.0,
        i_std_dev_i=0.0,
        i_std_dev_q=0.0,
    ):
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
        self._statistics_list = list()

    def add_data_block_statistic(self, i_data_block_statistic):
        if isinstance(i_data_block_statistic, DataBlockStatistic):
            self._statistics_list.append(i_data_block_statistic)
        else:
            raise TypeError("The input must be of type DataBlockStatistic")

    def get_data_block_statistic(self, index):
        if 0 <= index < self.get_number_of_data_block_statistic():
            return self._statistics_list[index]

        raise IndexError("DataBlockStatistic not available for index: {}".format(index))

    def get_number_of_data_block_statistic(self):
        return len(self._statistics_list)


class DataBlockStatistic(MetaDataElement):
    """
    DataBlockStatistic class
    """

    TYPE = "DataBlockStatistic"

    def __init__(
        self,
        i_line_start,
        i_line_stop,
        i_num_samples=0.0,
        i_max_i=0.0,
        i_max_q=0.0,
        i_min_i=0.0,
        i_min_q=0.0,
        i_sum_i=0.0,
        i_sum_q=0.0,
        i_sum_2_i=0.0,
        i_sum_2_q=0.0,
    ):
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
    """
    CoregPoly class
    """

    TYPE = "CoregPoly"

    class _CoregPoly1DAz(_Poly2D):
        TYPE = "_CoregPoly1DAz"

        _UNITS = 10 * ("",)

    class _CoregPoly1DRg(_Poly2D):
        TYPE = "_CoregPoly1DRg"

        _UNITS = 10 * ("",)

    def __init__(self, i_ref_az, i_ref_rg, i_coefficients_az, i_coefficients_rg):
        self._azimuth_coreg_poly = CoregPoly._CoregPoly1DAz(
            i_ref_az, i_ref_rg, i_coefficients_az
        )
        self._range_coreg_poly = CoregPoly._CoregPoly1DRg(
            i_ref_az, i_ref_rg, i_coefficients_rg
        )

    @property
    def ref_azimuth_time(self):
        return self._azimuth_coreg_poly.t_ref_az

    @ref_azimuth_time.setter
    def ref_azimuth_time(self, i_ref_az):
        self._azimuth_coreg_poly.t_ref_az = i_ref_az
        self._range_coreg_poly.t_ref_az = i_ref_az

    @property
    def ref_range_time(self):
        return self._azimuth_coreg_poly.t_ref_rg

    @ref_range_time.setter
    def ref_range_time(self, i_ref_rg):
        self._azimuth_coreg_poly.t_ref_rg = i_ref_rg
        self._range_coreg_poly.t_ref_rg = i_ref_rg

    @property
    def azimuth_poly(self):
        return self._azimuth_coreg_poly

    @property
    def range_poly(self):
        return self._range_coreg_poly


class CoregPolyVector(_Poly2DVector):
    """
    List of CoregPoly
    """

    TYPE = "CoregPolyVector"
    _SINGLE_POLY_TYPE = CoregPoly


class Pulse(MetaDataElement):
    """
    Pulse class
    """

    TYPE = "Pulse"

    def __init__(
        self,
        i_pulse_length=None,
        i_bandwidth=None,
        i_pulse_sampling_rate=None,
        i_pulse_energy=None,
        i_pulse_start_frequency=None,
        i_pulse_start_phase=None,
        i_pulse_direction=None,
    ):
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
    def pulse_direction(self):
        return self._pulse_direction

    @pulse_direction.setter
    def pulse_direction(self, i_pulse_direction):
        if i_pulse_direction is not None:
            self._pulse_direction = EPulseDirection(i_pulse_direction)
        else:
            self._pulse_direction = i_pulse_direction


class MetaDataChannel:
    """
    MetaDataChannel class
    """

    _supported_elements = [
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

    def __init__(self):
        self._contentID = None
        self._number = None
        self._total = None
        self._elements = collections.OrderedDict()
        for element_tag in self._supported_elements:
            self._elements[element_tag] = None
            setattr(self, element_tag, property(lambda s: s.get_element(element_tag)))

    @property
    def contentID(self) -> Optional[str]:
        return self._contentID

    @contentID.setter
    def contentID(self, ID: Optional[str]):
        self._contentID = ID

    @property
    def number(self) -> Optional[int]:
        return self._number

    @number.setter
    def number(self, number: Optional[int]):
        self._number = number

    @property
    def total(self) -> Optional[int]:
        return self._total

    @total.setter
    def total(self, total: Optional[int]):
        self._total = total

    def __repr__(self):
        str_repr = ["\nMetaDataChannel\n\n"]
        if self.contentID is not None:
            str_repr += ["ContentID={}\n".format(self.contentID)]

        str_repr += ["Number={}\n".format(self.number)]
        str_repr += ["Total={}\n".format(self.total)]

        str_repr += [str(e) for e in self._elements.values()]
        return "".join(str_repr)

    def insert_element(self, element: MetaDataElement) -> None:
        """Insert the specified metadata element.

        Parameters
        ----------
        element : MetaDataElement
            metadata element to be inserted

        Raises
        ------
        TypeError
            if provided metadata is not of type MetaDataElement
        """
        if isinstance(element, MetaDataElement):
            if self._elements[element.type()] is None:
                self._elements[element.type()] = element
            else:
                warnings.warn(
                    "The element {} is already present in the current metadata channel".format(
                        element.type()
                    )
                )
        else:
            raise TypeError

    def remove_element(self, element_type: str) -> None:
        """Remove specified channel element.

        Parameters
        ----------
        element_type : str
            element name

        Raises
        ------
        TypeError
            if specified element is not available
        """
        if element_type in MetaDataChannel.get_supported_metadata_elements():
            self._elements[element_type] = None
        else:
            raise TypeError

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
    def get_supported_metadata_elements() -> List[str]:
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
    """
    Metadata class

    List of MetaDataChannels
    """

    def __init__(self, description: str = ""):
        self.description = description
        self._metadatachannels = list()

    def append_channel(self, channel: MetaDataChannel) -> None:
        """Append the provided MetaDataChannel to the MetaData object.

        Parameters
        ----------
        channel : MetaDataChannel
            MetaDataChannel instance to be added

        Raises
        ------
        TypeError
            if input metadata object is not of type MetaDataChannel
        """
        if isinstance(channel, MetaDataChannel):
            self._metadatachannels.append(channel)
        else:
            raise TypeError

    def insert_element(
        self, element: MetaDataElement, meta_data_ch_index: int = 0
    ) -> None:
        """Inserting a new metadata element into the selected metadata channel.

        Parameters
        ----------
        element : MetaDataElement
            metadata element to be inserted
        meta_data_ch_index : int, optional
            metadata channel number where to insert, by default 0
        """
        self.get_metadata_channels(channel_index=meta_data_ch_index).insert_element(
            element
        )

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
        """Number of available channels in the metadata.

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

    def get_acquisition_time_line(
        self, meta_data_ch_index: int = 0
    ) -> AcquisitionTimeLine:
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
        return self.get_metadata_channels(
            meta_data_ch_index
        ).get_acquisition_time_line()

    def get_ground_corner_points(
        self, meta_data_ch_index: int = 0
    ) -> GroundCornerPoints:
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

    def get_doppler_centroid(
        self, meta_data_ch_index: int = 0
    ) -> DopplerCentroidVector:
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

    def get_tops_azimuth_modulation_rate(
        self, meta_data_ch_index: int = 0
    ) -> TopsAzimuthModulationRateVector:
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
        return self.get_metadata_channels(
            meta_data_ch_index
        ).get_tops_azimuth_modulation_rate()

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
