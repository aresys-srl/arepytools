# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Aresys metadata
---------------
"""
from dataclasses import dataclass, field
from typing import List, Optional

from arepytools.io.parsing.metadata_models.aresys_types import (
    AcquisitionTimelineType,
    AntennaInfoType,
    AttitudeInfoType,
    BurstInfoType,
    DataSetInfoType,
    DataStatisticsType,
    GroundCornersPointsType,
    PolyCoregType,
    PolyType,
    PulseType,
    RasterInfoType,
    SamplingConstantsType,
    StateVectorDataType,
    SwathInfoType,
    TreeElementBaseType,
)


@dataclass
class ChannelType(TreeElementBaseType):
    content_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "ContentID",
            "type": "Attribute",
        },
    )


@dataclass
class AresysXmlDocType:
    number_of_channels: Optional[int] = field(
        default=None,
        metadata={
            "name": "NumberOfChannels",
            "type": "Element",
            "required": True,
        },
    )
    version_number: Optional[float] = field(
        default=None,
        metadata={
            "name": "VersionNumber",
            "type": "Element",
            "required": True,
        },
    )
    description: Optional[str] = field(
        default=None,
        metadata={
            "name": "Description",
            "type": "Element",
            "required": True,
        },
    )
    channel: List["AresysXmlDocType.Channel"] = field(
        default_factory=list,
        metadata={
            "name": "Channel",
            "type": "Element",
        },
    )

    @dataclass
    class Channel(ChannelType):
        raster_info: Optional[RasterInfoType] = field(
            default=None,
            metadata={
                "name": "RasterInfo",
                "type": "Element",
            },
        )
        data_set_info: Optional[DataSetInfoType] = field(
            default=None,
            metadata={
                "name": "DataSetInfo",
                "type": "Element",
            },
        )
        swath_info: Optional[SwathInfoType] = field(
            default=None,
            metadata={
                "name": "SwathInfo",
                "type": "Element",
            },
        )
        sampling_constants: Optional[SamplingConstantsType] = field(
            default=None,
            metadata={
                "name": "SamplingConstants",
                "type": "Element",
            },
        )
        acquisition_time_line: Optional[AcquisitionTimelineType] = field(
            default=None,
            metadata={
                "name": "AcquisitionTimeLine",
                "type": "Element",
            },
        )
        data_statistics: Optional[DataStatisticsType] = field(
            default=None,
            metadata={
                "name": "DataStatistics",
                "type": "Element",
            },
        )
        burst_info: Optional[BurstInfoType] = field(
            default=None,
            metadata={
                "name": "BurstInfo",
                "type": "Element",
            },
        )
        state_vector_data: Optional[StateVectorDataType] = field(
            default=None,
            metadata={
                "name": "StateVectorData",
                "type": "Element",
            },
        )
        doppler_centroid: List[PolyType] = field(
            default_factory=list,
            metadata={
                "name": "DopplerCentroid",
                "type": "Element",
            },
        )
        doppler_rate: List[PolyType] = field(
            default_factory=list,
            metadata={
                "name": "DopplerRate",
                "type": "Element",
            },
        )
        tops_azimuth_modulation_rate: List[PolyType] = field(
            default_factory=list,
            metadata={
                "name": "TopsAzimuthModulationRate",
                "type": "Element",
            },
        )
        slant_to_ground: List[PolyType] = field(
            default_factory=list,
            metadata={
                "name": "SlantToGround",
                "type": "Element",
            },
        )
        ground_to_slant: List[PolyType] = field(
            default_factory=list,
            metadata={
                "name": "GroundToSlant",
                "type": "Element",
            },
        )
        slant_to_incidence: List[PolyType] = field(
            default_factory=list,
            metadata={
                "name": "SlantToIncidence",
                "type": "Element",
            },
        )
        slant_to_elevation: List[PolyType] = field(
            default_factory=list,
            metadata={
                "name": "SlantToElevation",
                "type": "Element",
            },
        )
        attitude_info: Optional[AttitudeInfoType] = field(
            default=None,
            metadata={
                "name": "AttitudeInfo",
                "type": "Element",
            },
        )
        ground_corner_points: Optional[GroundCornersPointsType] = field(
            default=None,
            metadata={
                "name": "GroundCornerPoints",
                "type": "Element",
            },
        )
        pulse: Optional[PulseType] = field(
            default=None,
            metadata={
                "name": "Pulse",
                "type": "Element",
            },
        )
        coreg_poly: List[PolyCoregType] = field(
            default_factory=list,
            metadata={
                "name": "CoregPoly",
                "type": "Element",
            },
        )
        antenna_info: Optional[AntennaInfoType] = field(
            default=None,
            metadata={
                "name": "AntennaInfo",
                "type": "Element",
            },
        )


@dataclass
class AresysXmlDoc(AresysXmlDocType):
    pass
