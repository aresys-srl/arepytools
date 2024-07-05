# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Aresys metadata types
---------------------
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

__NAMESPACE__ = "aresysTypes"


class AcquisitionModeType(Enum):
    NOT_SET = "NOT SET"
    STRIPMAP = "STRIPMAP"
    DOUBLE_POL = "DOUBLE POL"
    QUAD_POL = "QUAD POL"
    SCANSAR = "SCANSAR"
    TOPSAR = "TOPSAR"
    SPOT = "SPOT"
    WAVE = "WAVE"
    GMTI = "GMTI"


class AscendingDescendingType(Enum):
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class AttitudeType(Enum):
    NOMINAL = "NOMINAL"
    REFINED = "REFINED"


class CellTypeVerboseType(Enum):
    FLOAT_COMPLEX = "FLOAT_COMPLEX"
    FLOAT32 = "FLOAT32"
    DOUBLE_COMPLEX = "DOUBLE_COMPLEX"
    FLOAT64 = "FLOAT64"
    INT16 = "INT16"
    SHORT_COMPLEX = "SHORT_COMPLEX"
    INT32 = "INT32"
    INT_COMPLEX = "INT_COMPLEX"
    INT8 = "INT8"
    INT8_COMPLEX = "INT8_COMPLEX"
    CUSTOM = "CUSTOM"


@dataclass
class Dcomplex:
    class Meta:
        name = "DCOMPLEX"

    real_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "RealValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    imaginary_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "ImaginaryValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


class Endianity(Enum):
    BIGENDIAN = "BIGENDIAN"
    LITTLEENDIAN = "LITTLEENDIAN"


class GlobalPolarizationType(Enum):
    SINGLE_POL = "SINGLE POL"
    DUAL_POL = "DUAL POL"


class LeftRightType(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class PolarizationType(Enum):
    H_H = "H/H"
    H_V = "H/V"
    V_H = "V/H"
    V_V = "V/V"
    X_X = "X/X"


class PulseTypeDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"


class RasterFormatType(Enum):
    ARESYS_RASTER = "ARESYS_RASTER"
    ARESYS_GEOTIFF = "ARESYS_GEOTIFF"
    RASTER = "RASTER"


class ReferenceFrameType(Enum):
    GEOCENTRIC = "GEOCENTRIC"
    GEODETIC = "GEODETIC"
    ZERODOPPLER = "ZERODOPPLER"


class RotationOrderType(Enum):
    YPR = "YPR"
    YRP = "YRP"
    PRY = "PRY"
    PYR = "PYR"
    RPY = "RPY"
    RYP = "RYP"


class SensorNamesType(Enum):
    NOT_SET = "NOT SET"
    ASAR = "ASAR"
    PALSAR = "PALSAR"
    ERS1 = "ERS1"
    ERS2 = "ERS2"
    RADARSAT = "RADARSAT"
    TERRASARX = "TERRASARX"
    SENTINEL1 = "SENTINEL1"
    SENTINEL1_A = "SENTINEL1A"
    SENTINEL1_B = "SENTINEL1B"
    SENTINEL1_C = "SENTINEL1C"
    SENTINEL1_D = "SENTINEL1D"
    SAOCOM = "SAOCOM"
    SAOCOM_1_A = "SAOCOM-1A"
    SAOCOM_1_B = "SAOCOM-1B"
    UAVSAR = "UAVSAR"


@dataclass
class TreeElementBaseType:
    number: Optional[int] = field(
        default=None,
        metadata={
            "name": "Number",
            "type": "Attribute",
        },
    )
    total: Optional[int] = field(
        default=None,
        metadata={
            "name": "Total",
            "type": "Attribute",
        },
    )


class Units(Enum):
    VALUE = ""
    M = "m"
    S = "s"
    J = "j"
    D_B = "dB"
    RAD = "rad"
    DEG = "deg"
    M_S = "m/s"
    M_S2 = "m/s2"
    M_S3 = "m/s3"
    M_S4 = "m/s4"
    M_S5 = "m/s5"
    M_S6 = "m/s6"
    M_S7 = "m/s7"
    M_S8 = "m/s8"
    M_S9 = "m/s9"
    S_S = "s/s"
    S_S2 = "s/s2"
    S_S3 = "s/s3"
    S_S4 = "s/s4"
    S_S5 = "s/s5"
    HZ = "Hz"
    HZ_S = "Hz/s"
    HZ_S2 = "Hz/s2"
    HZ_S3 = "Hz/s3"
    HZ_S4 = "Hz/s4"
    HZ_S5 = "Hz/s5"
    HZ_S6 = "Hz/s6"
    HZ_S7 = "Hz/s7"
    HZ_S8 = "Hz/s8"
    HZ_S9 = "Hz/s9"
    RAD_S = "rad/s"
    RAD_S2 = "rad/s2"
    RAD_S3 = "rad/s3"
    RAD_S4 = "rad/s4"
    RAD_S5 = "rad/s5"
    RAD_S6 = "rad/s6"
    RAD_S7 = "rad/s7"
    RAD_S8 = "rad/s8"
    RAD_S9 = "rad/s9"
    S85 = "s85"
    UTC = "Utc"
    B = "b"
    K = "K"
    S_M = "s/m"
    S_M2 = "s/m2"
    S_M3 = "s/m3"
    S_M4 = "s/m4"
    S_M5 = "s/m5"
    S_M6 = "s/m6"
    S_M7 = "s/m7"
    S_M8 = "s/m8"
    S_M9 = "s/m9"
    DEG_S = "deg/s"
    DEG_S2 = "deg/s2"
    DEG_S3 = "deg/s3"
    DEG_S4 = "deg/s4"
    DEG_S5 = "deg/s5"
    DEG_S6 = "deg/s6"
    DEG_S7 = "deg/s7"
    DEG_S8 = "deg/s8"
    DEG_S9 = "deg/s9"


@dataclass
class AcquisitionTimelineType(TreeElementBaseType):
    """
    Acquisition timeline definition.

    :ivar missing_lines_number: Number of missing lines
    :ivar missing_lines_azimuthtimes: Azimuth relative times for each
        missing line
    :ivar duplicated_lines_number: Number of duplicated lines
    :ivar duplicated_lines_azimuthtimes: Azimuth relative times for each
        duplicated line
    :ivar prf_changes_number: Number of PRF changes
    :ivar prf_changes_azimuthtimes: Azimuth relative times for each PRF
        change
    :ivar prf_changes_values: PRF changes values
    :ivar swst_changes_number: Number of SWST changes
    :ivar swst_changes_azimuthtimes: Azimuth relative times for each
        SWST change
    :ivar swst_changes_values: SWST changes values
    :ivar noise_packets_number: Number of noise packets
    :ivar noise_packets_azimuthtimes: Azimuth relative times for each
        noise packet
    :ivar internal_calibration_number: Number of internal calibration
        packets
    :ivar internal_calibration_azimuthtimes: Azimuth relative times for
        each internal calibration packet
    :ivar swl_changes_number: Number of SWL changes
    :ivar swl_changes_azimuthtimes: Relative azimuth times for each SWL
        change
    :ivar swl_changes_values: SWL changes values
    :ivar chirp_period: Periodic list of chirp indexes related to multi-
        chirp image acquisition
    """

    missing_lines_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "MissingLines_number",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    missing_lines_azimuthtimes: Optional[
        "AcquisitionTimelineType.MissingLinesAzimuthtimes"
    ] = field(
        default=None,
        metadata={
            "name": "MissingLines_azimuthtimes",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    duplicated_lines_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "DuplicatedLines_number",
            "type": "Element",
            "namespace": "",
        },
    )
    duplicated_lines_azimuthtimes: Optional[
        "AcquisitionTimelineType.DuplicatedLinesAzimuthtimes"
    ] = field(
        default=None,
        metadata={
            "name": "DuplicatedLines_azimuthtimes",
            "type": "Element",
            "namespace": "",
        },
    )
    prf_changes_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "PRF_changes_number",
            "type": "Element",
            "namespace": "",
        },
    )
    prf_changes_azimuthtimes: Optional[
        "AcquisitionTimelineType.PrfChangesAzimuthtimes"
    ] = field(
        default=None,
        metadata={
            "name": "PRF_changes_azimuthtimes",
            "type": "Element",
            "namespace": "",
        },
    )
    prf_changes_values: Optional["AcquisitionTimelineType.PrfChangesValues"] = field(
        default=None,
        metadata={
            "name": "PRF_changes_values",
            "type": "Element",
            "namespace": "",
        },
    )
    swst_changes_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "Swst_changes_number",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swst_changes_azimuthtimes: Optional[
        "AcquisitionTimelineType.SwstChangesAzimuthtimes"
    ] = field(
        default=None,
        metadata={
            "name": "Swst_changes_azimuthtimes",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swst_changes_values: Optional["AcquisitionTimelineType.SwstChangesValues"] = field(
        default=None,
        metadata={
            "name": "Swst_changes_values",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    noise_packets_number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    noise_packets_azimuthtimes: Optional[
        "AcquisitionTimelineType.NoisePacketsAzimuthtimes"
    ] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_calibration_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "Internal_calibration_number",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_calibration_azimuthtimes: Optional[
        "AcquisitionTimelineType.InternalCalibrationAzimuthtimes"
    ] = field(
        default=None,
        metadata={
            "name": "Internal_calibration_azimuthtimes",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swl_changes_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "Swl_changes_number",
            "type": "Element",
            "namespace": "",
        },
    )
    swl_changes_azimuthtimes: Optional[
        "AcquisitionTimelineType.SwlChangesAzimuthtimes"
    ] = field(
        default=None,
        metadata={
            "name": "Swl_changes_azimuthtimes",
            "type": "Element",
            "namespace": "",
        },
    )
    swl_changes_values: Optional["AcquisitionTimelineType.SwlChangesValues"] = field(
        default=None,
        metadata={
            "name": "Swl_changes_values",
            "type": "Element",
            "namespace": "",
        },
    )
    chirp_period: Optional[str] = field(
        default=None,
        metadata={
            "name": "ChirpPeriod",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class MissingLinesAzimuthtimes:
        val: List["AcquisitionTimelineType.MissingLinesAzimuthtimes.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )

    @dataclass
    class DuplicatedLinesAzimuthtimes:
        val: List["AcquisitionTimelineType.DuplicatedLinesAzimuthtimes.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )

    @dataclass
    class PrfChangesAzimuthtimes:
        val: List["AcquisitionTimelineType.PrfChangesAzimuthtimes.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )

    @dataclass
    class PrfChangesValues:
        val: List["AcquisitionTimelineType.PrfChangesValues.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )

    @dataclass
    class SwstChangesAzimuthtimes:
        val: List["AcquisitionTimelineType.SwstChangesAzimuthtimes.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )

    @dataclass
    class SwstChangesValues:
        val: List["AcquisitionTimelineType.SwstChangesValues.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )

    @dataclass
    class NoisePacketsAzimuthtimes:
        val: List["AcquisitionTimelineType.NoisePacketsAzimuthtimes.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class InternalCalibrationAzimuthtimes:
        val: List["AcquisitionTimelineType.InternalCalibrationAzimuthtimes.Val"] = (
            field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "namespace": "",
                },
            )
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class SwlChangesAzimuthtimes:
        val: List["AcquisitionTimelineType.SwlChangesAzimuthtimes.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )

    @dataclass
    class SwlChangesValues:
        val: List["AcquisitionTimelineType.SwlChangesValues.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )


@dataclass
class AntennaInfoType(TreeElementBaseType):
    """
    Antenna pattern information.

    :ivar sensor_name: Sensor name: ASAR, PALSAR, ...
    :ivar acquisition_mode: Acquisition mode: STRIPMAP, TOPSAR, ...
    :ivar beam_name: Acquisition beam name
    :ivar polarization: Antenna polarization (H/H, H/V...)
    :ivar lines_per_pattern: Contains number of lines for each pattern
    """

    sensor_name: Optional[SensorNamesType] = field(
        default=None,
        metadata={
            "name": "SensorName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    acquisition_mode: Optional[AcquisitionModeType] = field(
        default=None,
        metadata={
            "name": "AcquisitionMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    beam_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "BeamName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarization: Optional[PolarizationType] = field(
        default=None,
        metadata={
            "name": "Polarization",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lines_per_pattern: Optional[int] = field(
        default=None,
        metadata={
            "name": "LinesPerPattern",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class AttitudeInfoType(TreeElementBaseType):
    """
    Sensor attitude information.

    :ivar t_ref_utc: Azimuth absolute start time for the first attitude
        value [Utc]
    :ivar dt_ypr_s: Azimuth time interval between two consecutive
        attitude values [s]
    :ivar n_ypr_n: Number of attitude values
    :ivar yaw_deg: Yaw angle values [deg]
    :ivar pitch_deg: Pitch angle values [deg]
    :ivar roll_deg: Roll angle values [deg]
    :ivar reference_frame: Reference frame: GEOCENTRIC, GEODETIC,
        ZERODOPPLER
    :ivar rotation_order: Rotation order: YPR, YRP, PRY, PYR, RPY, RYP
    :ivar attitude_type: Attitude type: NOMINAL, REFINED
    """

    t_ref_utc: Optional[str] = field(
        default=None,
        metadata={
            "name": "t_ref_Utc",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    dt_ypr_s: Optional["AttitudeInfoType.DtYprS"] = field(
        default=None,
        metadata={
            "name": "dtYPR_s",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    n_ypr_n: Optional["AttitudeInfoType.NYprN"] = field(
        default=None,
        metadata={
            "name": "nYPR_n",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    yaw_deg: Optional["AttitudeInfoType.YawDeg"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pitch_deg: Optional["AttitudeInfoType.PitchDeg"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    roll_deg: Optional["AttitudeInfoType.RollDeg"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_frame: Optional[ReferenceFrameType] = field(
        default=None,
        metadata={
            "name": "referenceFrame",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rotation_order: Optional[RotationOrderType] = field(
        default=None,
        metadata={
            "name": "rotationOrder",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    attitude_type: Optional[AttitudeType] = field(
        default=None,
        metadata={
            "name": "AttitudeType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class DtYprS:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class NYprN:
        value: Optional[int] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class YawDeg:
        val: List["AttitudeInfoType.YawDeg.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class PitchDeg:
        val: List["AttitudeInfoType.PitchDeg.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class RollDeg:
        val: List["AttitudeInfoType.RollDeg.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )


@dataclass
class DataBlockStatisticsType(TreeElementBaseType):
    """
    Statistics computed from data.

    :ivar num_samples: Number of samples analyzed
    :ivar max_i: Max of I (real) samples
    :ivar min_i: Min of I (real) samples
    :ivar max_q: Max of Q (imaginary) samples
    :ivar min_q: Min of Q (imaginary) samples
    :ivar sum_i: Sum of I (real) samples
    :ivar sum_q: Sum of Q (imaginary) samples
    :ivar sum2_i: Square Sum of I (real) samples
    :ivar sum2_q: Square Sum of Q (imaginary) samples
    :ivar line_start:
    :ivar line_stop:
    """

    num_samples: Optional["DataBlockStatisticsType.NumSamples"] = field(
        default=None,
        metadata={
            "name": "NumSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_i: Optional["DataBlockStatisticsType.MaxI"] = field(
        default=None,
        metadata={
            "name": "MaxI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    min_i: Optional["DataBlockStatisticsType.MinI"] = field(
        default=None,
        metadata={
            "name": "MinI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_q: Optional["DataBlockStatisticsType.MaxQ"] = field(
        default=None,
        metadata={
            "name": "MaxQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    min_q: Optional["DataBlockStatisticsType.MinQ"] = field(
        default=None,
        metadata={
            "name": "MinQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sum_i: Optional["DataBlockStatisticsType.SumI"] = field(
        default=None,
        metadata={
            "name": "SumI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sum_q: Optional["DataBlockStatisticsType.SumQ"] = field(
        default=None,
        metadata={
            "name": "SumQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sum2_i: Optional["DataBlockStatisticsType.Sum2I"] = field(
        default=None,
        metadata={
            "name": "Sum2I",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sum2_q: Optional["DataBlockStatisticsType.Sum2Q"] = field(
        default=None,
        metadata={
            "name": "Sum2Q",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    line_start: Optional[int] = field(
        default=None,
        metadata={
            "name": "lineStart",
            "type": "Attribute",
            "required": True,
        },
    )
    line_stop: Optional[int] = field(
        default=None,
        metadata={
            "name": "lineStop",
            "type": "Attribute",
            "required": True,
        },
    )

    @dataclass
    class NumSamples:
        value: Optional[int] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class MaxI:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class MinI:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class MaxQ:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class MinQ:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class SumI:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class SumQ:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class Sum2I:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class Sum2Q:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class DataSetInfoType(TreeElementBaseType):
    """
    Information regarding the dataset.

    :ivar sensor_name: Name of the sensor used to acquire the image:
        ASAR, PALSAR, ...
    :ivar description: Description of the image
    :ivar sense_date: Image acquisition date
    :ivar acquisition_mode: Image acquisition mode: STRIPMAP, TOPSAR,
        ...
    :ivar image_type: Image type: RAW DATA, RANGE FOCUSED, AZIMUTH
        FOCUSED
    :ivar projection: Image projection: SLANT RANGE, GROUND RANGE
    :ivar projection_parameters:
    :ivar acquisition_station: Image acquisition station
    :ivar processing_center: Image processing center
    :ivar processing_date: Image processing date
    :ivar processing_software: Image processing software
    :ivar fc_hz: Radar carrier frequency [Hz]
    :ivar side_looking: Radar side looking: LEFT, RIGHT
    :ivar external_calibration_factor: External calibration factor
    :ivar data_take_id:
    :ivar instrument_conf_id:
    """

    sensor_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "SensorName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    description: Optional["DataSetInfoType.Description"] = field(
        default=None,
        metadata={
            "name": "Description",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sense_date: Optional["DataSetInfoType.SenseDate"] = field(
        default=None,
        metadata={
            "name": "SenseDate",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    acquisition_mode: Optional["DataSetInfoType.AcquisitionMode"] = field(
        default=None,
        metadata={
            "name": "AcquisitionMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    image_type: Optional["DataSetInfoType.ImageType"] = field(
        default=None,
        metadata={
            "name": "ImageType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    projection: Optional["DataSetInfoType.Projection"] = field(
        default=None,
        metadata={
            "name": "Projection",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    projection_parameters: Optional["DataSetInfoType.ProjectionParameters"] = field(
        default=None,
        metadata={
            "name": "ProjectionParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    acquisition_station: Optional["DataSetInfoType.AcquisitionStation"] = field(
        default=None,
        metadata={
            "name": "AcquisitionStation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_center: Optional["DataSetInfoType.ProcessingCenter"] = field(
        default=None,
        metadata={
            "name": "ProcessingCenter",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_date: Optional["DataSetInfoType.ProcessingDate"] = field(
        default=None,
        metadata={
            "name": "ProcessingDate",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_software: Optional["DataSetInfoType.ProcessingSoftware"] = field(
        default=None,
        metadata={
            "name": "ProcessingSoftware",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    fc_hz: Optional["DataSetInfoType.FcHz"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    side_looking: Optional[LeftRightType] = field(
        default=None,
        metadata={
            "name": "SideLooking",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    external_calibration_factor: Optional[float] = field(
        default=None,
        metadata={
            "name": "ExternalCalibrationFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    data_take_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "DataTakeID",
            "type": "Element",
            "namespace": "",
        },
    )
    instrument_conf_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "InstrumentConfID",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 99999999,
        },
    )

    @dataclass
    class Description:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class SenseDate:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class AcquisitionMode:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        polarization: Optional[GlobalPolarizationType] = field(
            default=None,
            metadata={
                "name": "Polarization",
                "type": "Attribute",
            },
        )

    @dataclass
    class ImageType:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class Projection:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class ProjectionParameters:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        format: Optional[str] = field(
            default=None,
            metadata={
                "name": "Format",
                "type": "Attribute",
                "required": True,
            },
        )

    @dataclass
    class AcquisitionStation:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class ProcessingCenter:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class ProcessingDate:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class ProcessingSoftware:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class FcHz:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class PointType:
    val: List["PointType.Val"] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "",
            "min_occurs": 5,
            "max_occurs": 5,
        },
    )

    @dataclass
    class Val:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class SamplingConstantsType(TreeElementBaseType):
    """
    Bandwidths and sampling frequencies.

    :ivar frg_hz: Range sampling frequency [Hz]
    :ivar brg_hz: Range bandwidth [Hz]
    :ivar faz_hz: Azimuth sampling frequency [Hz]
    :ivar baz_hz: Azimuth bandwidth [Hz]
    """

    frg_hz: Optional["SamplingConstantsType.FrgHz"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    brg_hz: Optional["SamplingConstantsType.BrgHz"] = field(
        default=None,
        metadata={
            "name": "Brg_hz",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    faz_hz: Optional["SamplingConstantsType.FazHz"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    baz_hz: Optional["SamplingConstantsType.BazHz"] = field(
        default=None,
        metadata={
            "name": "Baz_hz",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class FrgHz:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class BrgHz:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class FazHz:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class BazHz:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class StateVectorDataType(TreeElementBaseType):
    """
    Information regarding position and velocity of the sensor along the orbit
    (State Vector Data)

    :ivar orbit_number: Number of the orbit
    :ivar track: Number of the track
    :ivar orbit_direction: Direction of the orbit: ASCENDING, DESCENDING
    :ivar p_sv_m: Orbit state vectors position coordinates (xyz) [m]
    :ivar v_sv_m_os: Orbit state vectors velocity coordinates [m/s]
    :ivar t_ref_utc: Azimuth absolute start time for the first state
        vector [Utc]
    :ivar dt_sv_s: Azimuth time interval between two consecutive state
        vectors [s]
    :ivar n_sv_n: Number of state vectors
    :ivar ascending_node_time: Azimuth absolute time of the ascending
        node
    :ivar ascending_node_coords: Coordinates of the ascending node
    """

    orbit_number: Optional[str] = field(
        default=None,
        metadata={
            "name": "OrbitNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    track: Optional[str] = field(
        default=None,
        metadata={
            "name": "Track",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    orbit_direction: Optional[AscendingDescendingType] = field(
        default=None,
        metadata={
            "name": "OrbitDirection",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    p_sv_m: Optional["StateVectorDataType.PSvM"] = field(
        default=None,
        metadata={
            "name": "pSV_m",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    v_sv_m_os: Optional["StateVectorDataType.VSvMOs"] = field(
        default=None,
        metadata={
            "name": "vSV_mOs",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    t_ref_utc: Optional[str] = field(
        default=None,
        metadata={
            "name": "t_ref_Utc",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    dt_sv_s: Optional["StateVectorDataType.DtSvS"] = field(
        default=None,
        metadata={
            "name": "dtSV_s",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    n_sv_n: Optional["StateVectorDataType.NSvN"] = field(
        default=None,
        metadata={
            "name": "nSV_n",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ascending_node_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "AscendingNodeTime",
            "type": "Element",
            "namespace": "",
        },
    )
    ascending_node_coords: Optional["StateVectorDataType.AscendingNodeCoords"] = field(
        default=None,
        metadata={
            "name": "AscendingNodeCoords",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class PSvM:
        val: List["StateVectorDataType.PSvM.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class VSvMOs:
        val: List["StateVectorDataType.VSvMOs.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class DtSvS:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class NSvN:
        value: Optional[int] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class AscendingNodeCoords:
        val: List["StateVectorDataType.AscendingNodeCoords.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 3,
                "max_occurs": 3,
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )


@dataclass
class DoubleWithUnit:
    class Meta:
        name = "doubleWithUnit"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: Optional[Units] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class PolyCoregType(TreeElementBaseType):
    """
    Polynomial parametrization of the coregistration parameters.

    :ivar pol_rg: Polynomial coefficients of Range
    :ivar pol_az: Polynomial coefficients of Azimuth
    :ivar trg0_s: Polynomial range reference time [s]
    :ivar taz0_utc: Polynomial azimuth reference time [Utc]
    """

    class Meta:
        name = "polyCoregType"

    pol_rg: Optional["PolyCoregType.PolRg"] = field(
        default=None,
        metadata={
            "name": "polRg",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pol_az: Optional["PolyCoregType.PolAz"] = field(
        default=None,
        metadata={
            "name": "polAz",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    trg0_s: Optional["PolyCoregType.Trg0S"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    taz0_utc: Optional["PolyCoregType.Taz0Utc"] = field(
        default=None,
        metadata={
            "name": "taz0_Utc",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class PolRg:
        val: List["PolyCoregType.PolRg.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 4,
                "max_occurs": 4,
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class PolAz:
        val: List["PolyCoregType.PolAz.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 4,
                "max_occurs": 4,
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class Trg0S:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class Taz0Utc:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class PolyType(TreeElementBaseType):
    """
    Polynomial parametrization of the geometry parameter.

    :ivar pol: Polynomial coefficients: const, rg, az, az*rg, rg^2,
        rg^3, rg^4 [Optional: rg^5 rg^6 .... rg^N]
    :ivar trg0_s: Polynomial range reference time [s]
    :ivar taz0_utc: Polynomial azimuth reference time [Utc]
    """

    class Meta:
        name = "polyType"

    pol: Optional["PolyType.Pol"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    trg0_s: Optional["PolyType.Trg0S"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    taz0_utc: Optional["PolyType.Taz0Utc"] = field(
        default=None,
        metadata={
            "name": "taz0_Utc",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class Pol:
        val: List["PolyType.Pol.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 7,
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class Trg0S:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class Taz0Utc:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class StringWithUnit:
    class Meta:
        name = "stringWithUnit"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    unit: Optional[Units] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class BurstType:
    """
    :ivar range_start_time: Range absolute start time [s]
    :ivar azimuth_start_time: Azimuth start time absolute value [Utc]
    :ivar burst_center_azimuth_shift:
    :ivar n:
    """

    range_start_time: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "RangeStartTime",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_start_time: Optional[StringWithUnit] = field(
        default=None,
        metadata={
            "name": "AzimuthStartTime",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    burst_center_azimuth_shift: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "BurstCenterAzimuthShift",
            "type": "Element",
            "namespace": "",
        },
    )
    n: Optional[int] = field(
        default=None,
        metadata={
            "name": "N",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class DataStatisticsType(TreeElementBaseType):
    """
    Statistics computed from data.

    :ivar num_samples: Number of samples analyzed
    :ivar max_i: Max of I (real) samples
    :ivar min_i: Min of I (real) samples
    :ivar max_q: Max of Q (imaginary) samples
    :ivar min_q: Min of Q (imaginary) samples
    :ivar sum_i: Sum of I (real) samples
    :ivar sum_q: Sum of Q (imaginary) samples
    :ivar sum2_i: Square Sum of I (real) samples
    :ivar sum2_q: Square Sum of Q (imaginary) samples
    :ivar std_dev_i: Standard Deviation of I (real) samples
    :ivar std_dev_q: Standard Deviation of Q (imaginary) samples
    :ivar statistics_list:
    """

    num_samples: Optional["DataStatisticsType.NumSamples"] = field(
        default=None,
        metadata={
            "name": "NumSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_i: Optional["DataStatisticsType.MaxI"] = field(
        default=None,
        metadata={
            "name": "MaxI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    min_i: Optional["DataStatisticsType.MinI"] = field(
        default=None,
        metadata={
            "name": "MinI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_q: Optional["DataStatisticsType.MaxQ"] = field(
        default=None,
        metadata={
            "name": "MaxQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    min_q: Optional["DataStatisticsType.MinQ"] = field(
        default=None,
        metadata={
            "name": "MinQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sum_i: Optional["DataStatisticsType.SumI"] = field(
        default=None,
        metadata={
            "name": "SumI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sum_q: Optional["DataStatisticsType.SumQ"] = field(
        default=None,
        metadata={
            "name": "SumQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sum2_i: Optional["DataStatisticsType.Sum2I"] = field(
        default=None,
        metadata={
            "name": "Sum2I",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sum2_q: Optional["DataStatisticsType.Sum2Q"] = field(
        default=None,
        metadata={
            "name": "Sum2Q",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    std_dev_i: Optional["DataStatisticsType.StdDevI"] = field(
        default=None,
        metadata={
            "name": "StdDevI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    std_dev_q: Optional["DataStatisticsType.StdDevQ"] = field(
        default=None,
        metadata={
            "name": "StdDevQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    statistics_list: Optional["DataStatisticsType.StatisticsList"] = field(
        default=None,
        metadata={
            "name": "StatisticsList",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class NumSamples:
        value: Optional[int] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class MaxI:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class MinI:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class MaxQ:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class MinQ:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class SumI:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class SumQ:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class Sum2I:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class Sum2Q:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class StdDevI:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class StdDevQ:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class StatisticsList:
        data_block_statistic: List[DataBlockStatisticsType] = field(
            default_factory=list,
            metadata={
                "name": "DataBlockStatistic",
                "type": "Element",
                "namespace": "",
                "min_occurs": 1,
            },
        )


@dataclass
class GroundCornersPointsType(TreeElementBaseType):
    easting_grid_size: Optional["GroundCornersPointsType.EastingGridSize"] = field(
        default=None,
        metadata={
            "name": "EastingGridSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    northing_grid_size: Optional["GroundCornersPointsType.NorthingGridSize"] = field(
        default=None,
        metadata={
            "name": "NorthingGridSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    north_west: Optional["GroundCornersPointsType.NorthWest"] = field(
        default=None,
        metadata={
            "name": "NorthWest",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    north_east: Optional["GroundCornersPointsType.NorthEast"] = field(
        default=None,
        metadata={
            "name": "NorthEast",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    south_west: Optional["GroundCornersPointsType.SouthWest"] = field(
        default=None,
        metadata={
            "name": "SouthWest",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    south_east: Optional["GroundCornersPointsType.SouthEast"] = field(
        default=None,
        metadata={
            "name": "SouthEast",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    center: Optional["GroundCornersPointsType.Center"] = field(
        default=None,
        metadata={
            "name": "Center",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class EastingGridSize:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class NorthingGridSize:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class NorthWest:
        point: Optional[PointType] = field(
            default=None,
            metadata={
                "name": "Point",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

    @dataclass
    class NorthEast:
        point: Optional[PointType] = field(
            default=None,
            metadata={
                "name": "Point",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

    @dataclass
    class SouthWest:
        point: Optional[PointType] = field(
            default=None,
            metadata={
                "name": "Point",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

    @dataclass
    class SouthEast:
        point: Optional[PointType] = field(
            default=None,
            metadata={
                "name": "Point",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

    @dataclass
    class Center:
        point: Optional[PointType] = field(
            default=None,
            metadata={
                "name": "Point",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )


@dataclass
class PulseType(TreeElementBaseType):
    """
    Transmitted pulse parameters.

    :ivar direction: Pulse direction (UP, DOWN)
    :ivar pulse_length: Pulse length [s]
    :ivar bandwidth: Pulse bandwidth [Hz]
    :ivar pulse_energy: Pulse energy [J]
    :ivar pulse_sampling_rate: Pulse sampling rate [Hz]
    :ivar pulse_start_frequency: Pulse start frequency [Hz]
    :ivar pulse_start_phase: Pulse start phase [rad]
    """

    direction: Optional[PulseTypeDirection] = field(
        default=None,
        metadata={
            "name": "Direction",
            "type": "Element",
            "namespace": "",
        },
    )
    pulse_length: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "PulseLength",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    bandwidth: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "Bandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pulse_energy: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "PulseEnergy",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pulse_sampling_rate: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "PulseSamplingRate",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pulse_start_frequency: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "PulseStartFrequency",
            "type": "Element",
            "namespace": "",
        },
    )
    pulse_start_phase: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "PulseStartPhase",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class RasterInfoType(TreeElementBaseType):
    """
    Information regarding binary file format and time coordinates of the image.

    :ivar file_name: Name of the associated binary file
    :ivar lines: Total number of lines (azimuth) of the image
    :ivar samples: Total number of samples (range) of the image
    :ivar header_offset_bytes: Number of bytes at the beginning of the
        file containing the header information
    :ivar row_prefix_bytes: Number of bytes at the beginning of each
        line containing header information
    :ivar byte_order: Endianity: BIGENDIAN or LITTLEENDIAN
    :ivar cell_type: Byte format type: FLOAT_COMPLEX, FLOAT32,
        DOUBLE_COMPLEX, FLOAT64, INT16, SHORT_COMPLEX, INT32,
        INT_COMPLEX, INT8, INT8_COMPLEX
    :ivar lines_step: Azimuth sampling step [s]
    :ivar samples_step: Range sampling step [s]
    :ivar lines_start: Azimuth absolute start time [Utc]
    :ivar samples_start: Range absolute start time [s]
    :ivar raster_format: Raster Format of the Data ( Default value is
        ARESYS_RASTER )
    :ivar invalid_value:
    """

    file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "FileName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lines: Optional[int] = field(
        default=None,
        metadata={
            "name": "Lines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    samples: Optional[int] = field(
        default=None,
        metadata={
            "name": "Samples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    header_offset_bytes: Optional[int] = field(
        default=None,
        metadata={
            "name": "HeaderOffsetBytes",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    row_prefix_bytes: Optional[int] = field(
        default=None,
        metadata={
            "name": "RowPrefixBytes",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    byte_order: Optional[Endianity] = field(
        default=None,
        metadata={
            "name": "ByteOrder",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    cell_type: Optional[CellTypeVerboseType] = field(
        default=None,
        metadata={
            "name": "CellType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lines_step: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "LinesStep",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    samples_step: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "SamplesStep",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lines_start: Optional[StringWithUnit] = field(
        default=None,
        metadata={
            "name": "LinesStart",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    samples_start: Optional[StringWithUnit] = field(
        default=None,
        metadata={
            "name": "SamplesStart",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raster_format: Optional[RasterFormatType] = field(
        default=None,
        metadata={
            "name": "RasterFormat",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_value: Optional[Dcomplex] = field(
        default=None,
        metadata={
            "name": "InvalidValue",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class SwathInfoType(TreeElementBaseType):
    """
    Swath general information.

    :ivar swath: Swath name
    :ivar swath_acquisition_order: Swath acquisition order
    :ivar polarization: Polarization: H/H, H/V, V/H, V/V
    :ivar rank: Rank
    :ivar range_delay_bias: Range delay bias [s]
    :ivar acquisition_start_time: Acquisition start time [Utc]
    :ivar azimuth_steering_angle_reference_time: Azimuth antenna steering
        polynomial reference time [s]
    :ivar azimuth_steering_angle_pol: Azimuth antenna steering polynomial
        coefficients: const [rad], az [rad/s], az^2 [rad/s^2], az^3
        [rad/s^3]
    :ivar azimuth_steering_rate_reference_time: Azimuth antenna steering
        rate polynomial reference time [s]
    :ivar azimuth_steering_rate_pol: Azimuth antenna steering rate
        polynomial coefficients: const [rad/s], az [rad/s^2], az^2
        [rad/s^3]
    :ivar acquisition_prf: Acquisition Pulse Repetition Frequency
    :ivar echoes_per_burst: Number of echoes for each burst
    :ivar channel_delay: Range channel delay time
    :ivar rx_gain: Value of the commandable Rx attenuation in the
        receiver channel
    """

    swath: Optional["SwathInfoType.Swath"] = field(
        default=None,
        metadata={
            "name": "Swath",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swath_acquisition_order: Optional["SwathInfoType.SwathAcquisitionOrder"] = field(
        default=None,
        metadata={
            "name": "SwathAcquisitionOrder",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarization: Optional[PolarizationType] = field(
        default=None,
        metadata={
            "name": "Polarization",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rank: Optional["SwathInfoType.Rank"] = field(
        default=None,
        metadata={
            "name": "Rank",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_delay_bias: Optional["SwathInfoType.RangeDelayBias"] = field(
        default=None,
        metadata={
            "name": "RangeDelayBias",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    acquisition_start_time: Optional["SwathInfoType.AcquisitionStartTime"] = field(
        default=None,
        metadata={
            "name": "AcquisitionStartTime",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_steering_angle_reference_time: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "AzimuthSteeringAngleReferenceTime",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_steering_angle_pol: Optional["SwathInfoType.AzimuthSteeringAnglePol"] = (
        field(
            default=None,
            metadata={
                "name": "AzimuthSteeringAnglePol",
                "type": "Element",
                "namespace": "",
            },
        )
    )
    azimuth_steering_rate_reference_time: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "AzimuthSteeringRateReferenceTime",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_steering_rate_pol: Optional["SwathInfoType.AzimuthSteeringRatePol"] = field(
        default=None,
        metadata={
            "name": "AzimuthSteeringRatePol",
            "type": "Element",
            "namespace": "",
        },
    )
    acquisition_prf: Optional[float] = field(
        default=None,
        metadata={
            "name": "AcquisitionPRF",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    echoes_per_burst: Optional[int] = field(
        default=None,
        metadata={
            "name": "EchoesPerBurst",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    channel_delay: Optional[float] = field(
        default=None,
        metadata={
            "name": "ChannelDelay",
            "type": "Element",
            "namespace": "",
        },
    )
    rx_gain: Optional[float] = field(
        default=None,
        metadata={
            "name": "RxGain",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class Swath:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class SwathAcquisitionOrder:
        value: Optional[int] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class Rank:
        value: Optional[int] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class RangeDelayBias:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class AcquisitionStartTime:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: Optional[Units] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class AzimuthSteeringAnglePol:
        val: List["SwathInfoType.AzimuthSteeringAnglePol.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 4,
                "max_occurs": 4,
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )

    @dataclass
    class AzimuthSteeringRatePol:
        val: List["SwathInfoType.AzimuthSteeringRatePol.Val"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 3,
                "max_occurs": 3,
            },
        )

        @dataclass
        class Val:
            value: Optional[float] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            n: Optional[int] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Attribute",
                },
            )
            unit: Optional[Units] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                },
            )


@dataclass
class BurstInfoType(TreeElementBaseType):
    """
    Bursts information.

    :ivar number_of_bursts: Number of bursts in the swath
    :ivar lines_per_burst: Number of lines in each burst
    :ivar lines_per_burst_change_list:
    :ivar burst_repetition_frequency: Burst repetition frequency [Hz]
    :ivar burst: Time coordinates of each burst
    """

    number_of_bursts: Optional[int] = field(
        default=None,
        metadata={
            "name": "NumberOfBursts",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lines_per_burst: Optional[int] = field(
        default=None,
        metadata={
            "name": "LinesPerBurst",
            "type": "Element",
            "namespace": "",
        },
    )
    lines_per_burst_change_list: Optional["BurstInfoType.LinesPerBurstChangeList"] = (
        field(
            default=None,
            metadata={
                "name": "LinesPerBurstChangeList",
                "type": "Element",
                "namespace": "",
            },
        )
    )
    burst_repetition_frequency: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "BurstRepetitionFrequency",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    burst: List[BurstType] = field(
        default_factory=list,
        metadata={
            "name": "Burst",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )

    @dataclass
    class LinesPerBurstChangeList:
        lines: List["BurstInfoType.LinesPerBurstChangeList.Lines"] = field(
            default_factory=list,
            metadata={
                "name": "Lines",
                "type": "Element",
                "namespace": "",
                "min_occurs": 1,
            },
        )

        @dataclass
        class Lines:
            value: Optional[int] = field(
                default=None,
                metadata={
                    "required": True,
                },
            )
            from_burst: Optional[int] = field(
                default=None,
                metadata={
                    "name": "FromBurst",
                    "type": "Attribute",
                    "required": True,
                },
            )
