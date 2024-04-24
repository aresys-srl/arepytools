# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Translate
---------
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from arepytools.io import metadata
from arepytools.io.parsing import metadata_models
from arepytools.timing.precisedatetime import PreciseDateTime


def translate_endianity_from_model(
    endianity: metadata_models.Endianity,
) -> metadata.EByteOrder:
    """Translate endianity from model"""
    return metadata.EByteOrder(endianity.value)


def translate_endianity_to_model(
    endianity: metadata.EByteOrder,
) -> metadata_models.Endianity:
    """Translate endianity to model"""
    return metadata_models.Endianity(endianity.value)


def translate_cell_type_from_model(
    cell_type: metadata_models.CellTypeVerboseType,
) -> metadata.ECellType:
    """Translate cell type from model"""
    if cell_type == metadata_models.CellTypeVerboseType.SHORT_COMPLEX:
        return metadata.ECellType.i16complex

    if cell_type == metadata_models.CellTypeVerboseType.INT_COMPLEX:
        return metadata.ECellType.i32complex

    return metadata.ECellType(cell_type.value)


def translate_cell_type_to_model(
    cell_type: metadata.ECellType,
) -> metadata_models.CellTypeVerboseType:
    """Translate cell type to model"""
    if cell_type == metadata.ECellType.i16complex:
        return metadata_models.CellTypeVerboseType.SHORT_COMPLEX

    if cell_type == metadata.ECellType.i32complex:
        return metadata_models.CellTypeVerboseType.INT_COMPLEX

    return metadata_models.CellTypeVerboseType(cell_type.value)


def translate_orbit_direction_from_model(
    direction: metadata_models.AscendingDescendingType,
) -> Optional[metadata.EOrbitDirection]:
    """Translate orbit direction from model"""
    if direction == metadata_models.AscendingDescendingType.NOT_AVAILABLE:
        return None

    return metadata.EOrbitDirection(direction.value)


def translate_orbit_direction_to_model(
    direction: Optional[metadata.EOrbitDirection],
) -> metadata_models.AscendingDescendingType:
    """Translate orbit direction to model"""
    if direction is not None:
        return metadata_models.AscendingDescendingType(direction.value)
    return metadata_models.AscendingDescendingType.NOT_AVAILABLE


def translate_side_looking_from_model(
    side: metadata_models.LeftRightType,
) -> metadata.ESideLooking:
    """Translate looking side from model"""
    return metadata.ESideLooking(side.value)


def translate_side_looking_to_model(
    side: metadata.ESideLooking,
) -> metadata_models.LeftRightType:
    """Translate looking side to model"""
    return metadata_models.LeftRightType(side.value)


def translate_polarization_from_model(
    polarization: metadata_models.PolarizationType,
) -> metadata.EPolarization:
    """Translate polarization from model"""
    return metadata.EPolarization(polarization.value)


def translate_polarization_to_model(
    polarization: metadata.EPolarization,
) -> metadata_models.PolarizationType:
    """Translate polarization from model"""
    unsupported_polarizations = {
        metadata.EPolarization.none,
        metadata.EPolarization.crh,
        metadata.EPolarization.crv,
        metadata.EPolarization.clh,
        metadata.EPolarization.clv,
        metadata.EPolarization.ch,
        metadata.EPolarization.cv,
        metadata.EPolarization.xh,
        metadata.EPolarization.xv,
        metadata.EPolarization.hx,
        metadata.EPolarization.vx,
    }

    if polarization in unsupported_polarizations:
        raise RuntimeError(f"Polarization {polarization} is not supported")

    return metadata_models.PolarizationType(polarization.value)


def translate_reference_frame_from_model(
    frame: metadata_models.ReferenceFrameType,
) -> metadata.EReferenceFrame:
    """Translate reference frame from model"""
    return metadata.EReferenceFrame(frame.value)


def translate_reference_frame_to_model(
    frame: metadata.EReferenceFrame,
) -> metadata_models.ReferenceFrameType:
    """Translate reference frame to model"""
    if frame == metadata.EReferenceFrame.none:
        raise RuntimeError(f"Reference frame {frame} is not supported")

    return metadata_models.ReferenceFrameType(frame.value)


def translate_rotation_order_from_model(
    order: metadata_models.RotationOrderType,
) -> metadata.ERotationOrder:
    """Translate rotation order from model"""
    return metadata.ERotationOrder(order.value.lower())


def translate_rotation_order_to_model(
    order: metadata.ERotationOrder,
) -> metadata_models.RotationOrderType:
    """Translate rotation order to model"""
    if order == metadata.ERotationOrder.none:
        raise RuntimeError(f"Rotation order {order} is not supported")

    return metadata_models.RotationOrderType(order.value.upper())


def translate_attitude_type_from_model(
    attitude_type: metadata_models.AttitudeType,
) -> metadata.EAttitudeType:
    """Translate attitude type from model"""
    return metadata.EAttitudeType(attitude_type.value)


def translate_attitude_type_to_model(
    attitude_type: metadata.EAttitudeType,
) -> metadata_models.AttitudeType:
    """Translate attitude type to model"""
    if attitude_type == metadata.EAttitudeType.none:
        raise RuntimeError(f"Attitude type {attitude_type} is not supported")

    return metadata_models.AttitudeType(attitude_type.value)


def translate_raster_format_type_from_model(
    raster_format: metadata_models.RasterFormatType,
) -> metadata.ERasterFormatType:
    """Translate raster format type from model"""
    if raster_format == metadata_models.RasterFormatType.RASTER:
        return metadata.ERasterFormatType.aresys_raster
    return metadata.ERasterFormatType(raster_format.value)


def translate_raster_format_type_to_model(
    raster_format: metadata.ERasterFormatType,
) -> metadata_models.RasterFormatType:
    """Translate raster format type to model"""
    if raster_format == metadata.ERasterFormatType.raster:
        return metadata_models.RasterFormatType.ARESYS_RASTER
    return metadata_models.RasterFormatType(raster_format.value)


def translate_unit_from_model(unit: metadata_models.Units) -> str:
    """Translate unit from model"""
    return unit.value


def translate_unit_to_model(unit: str) -> metadata_models.Units:
    """Translate unit to model"""
    return metadata_models.Units(unit)


def translate_double_with_unit_to_model(
    value: float, unit: str
) -> metadata_models.DoubleWithUnit:
    """Create a DoubleWithUnit model"""
    return metadata_models.DoubleWithUnit(
        value=value, unit=translate_unit_to_model(unit)
    )


def translate_str_with_unit_from_model(
    str_with_unit: metadata_models.StringWithUnit,
) -> Tuple[Union[PreciseDateTime, float], str]:
    """Translate a StringWithUnit model"""
    assert str_with_unit.unit is not None

    if str_with_unit.unit == metadata_models.Units.UTC:
        value = PreciseDateTime.from_utc_string(str_with_unit.value)
    else:
        value = float(str_with_unit.value)

    return value, translate_unit_from_model(str_with_unit.unit)


def translate_str_with_unit_to_model(
    value: Union[PreciseDateTime, float], unit: str
) -> metadata_models.StringWithUnit:
    """Translate a StringWithUnit from model"""
    return metadata_models.StringWithUnit(
        value=str(value), unit=translate_unit_to_model(unit)
    )


def translate_dcomplex_to_model(value: complex) -> metadata_models.Dcomplex:
    """Translate dcomplex to model"""
    return metadata_models.Dcomplex(real_value=value.real, imaginary_value=value.imag)


def translate_dcomplex_from_model(value: metadata_models.Dcomplex) -> complex:
    """Translate dcomplex from model"""
    assert value.real_value is not None and value.imaginary_value is not None
    return complex(real=value.real_value, imag=value.imaginary_value)


def translate_raster_info_to_model(
    raster_info: metadata.RasterInfo,
) -> metadata_models.RasterInfoType:
    """Translate raster info to model"""
    return metadata_models.RasterInfoType(
        number=None,
        total=None,
        file_name=raster_info.file_name,
        lines=raster_info.lines,
        samples=raster_info.samples,
        header_offset_bytes=raster_info.header_offset_bytes,
        row_prefix_bytes=raster_info.row_prefix_bytes,
        byte_order=translate_endianity_to_model(raster_info.byte_order),
        cell_type=translate_cell_type_to_model(raster_info.cell_type),
        lines_step=translate_double_with_unit_to_model(
            raster_info.lines_step, raster_info.lines_step_unit
        ),
        samples_step=translate_double_with_unit_to_model(
            raster_info.samples_step, raster_info.samples_step_unit
        ),
        lines_start=translate_str_with_unit_to_model(
            raster_info.lines_start, unit=raster_info.lines_start_unit
        ),
        samples_start=translate_str_with_unit_to_model(
            raster_info.samples_start, unit=raster_info.samples_start_unit
        ),
        invalid_value=(
            translate_dcomplex_to_model(raster_info.invalid_value)
            if raster_info.invalid_value is not None
            else None
        ),
        raster_format=(
            translate_raster_format_type_to_model(raster_info.format_type)
            if raster_info.format_type is not None
            else None
        ),
    )


def translate_raster_info_from_model(
    raster_info: metadata_models.RasterInfoType,
) -> metadata.RasterInfo:
    """Translate raster info from model"""
    assert raster_info.file_name is not None
    assert raster_info.lines is not None
    assert raster_info.samples is not None
    assert raster_info.header_offset_bytes is not None
    assert raster_info.row_prefix_bytes is not None
    assert raster_info.byte_order is not None
    assert raster_info.cell_type is not None
    assert (
        raster_info.lines_step is not None and raster_info.lines_step.value is not None
    )
    assert (
        raster_info.samples_step is not None
        and raster_info.samples_step.value is not None
    )
    assert raster_info.lines_start is not None
    assert raster_info.samples_start is not None

    raster_info_out = metadata.RasterInfo(
        filename=raster_info.file_name,
        lines=raster_info.lines,
        samples=raster_info.samples,
        header_offset_bytes=raster_info.header_offset_bytes,
        row_prefix_bytes=raster_info.row_prefix_bytes,
        byteorder=translate_endianity_from_model(raster_info.byte_order),
        celltype=translate_cell_type_from_model(raster_info.cell_type),
        invalid_value=(
            translate_dcomplex_from_model(raster_info.invalid_value)
            if raster_info.invalid_value is not None
            else None
        ),
        format_type=(
            translate_raster_format_type_from_model(raster_info.raster_format)
            if raster_info.raster_format is not None
            else None
        ),
    )

    assert raster_info.lines_step.unit is not None
    lines_start, lines_start_unit = translate_str_with_unit_from_model(
        raster_info.lines_start
    )
    raster_info_out.set_lines_axis(
        lines_start,
        lines_start_unit,
        raster_info.lines_step.value,
        translate_unit_from_model(raster_info.lines_step.unit),
    )

    assert raster_info.samples_step.unit is not None
    samples_start, samples_start_unit = translate_str_with_unit_from_model(
        raster_info.samples_start
    )
    raster_info_out.set_samples_axis(
        samples_start,
        samples_start_unit,
        raster_info.samples_step.value,
        translate_unit_from_model(raster_info.samples_step.unit),
    )

    return raster_info_out


def translate_dataset_info_from_model(
    info: metadata_models.DataSetInfoType,
) -> metadata.DataSetInfo:
    """Translate dataset info from model"""
    assert info.sensor_name is not None
    assert info.description is not None
    assert info.sense_date is not None
    assert info.acquisition_mode is not None
    assert info.image_type is not None
    assert info.projection is not None
    assert info.acquisition_station is not None
    assert info.processing_center is not None
    assert info.processing_date is not None
    assert info.processing_software is not None
    assert info.fc_hz is not None
    assert info.fc_hz.value is not None
    assert info.side_looking is not None

    info_out = metadata.DataSetInfo(
        acquisition_mode_i=info.acquisition_mode.value, fc_hz_i=info.fc_hz.value
    )

    if info.sense_date.value != "NOT_AVAILABLE":
        info_out.sense_date = PreciseDateTime.from_utc_string(info.sense_date.value)
    if info.processing_date.value != "NOT_AVAILABLE":
        info_out.processing_date = PreciseDateTime.from_utc_string(
            info.processing_date.value
        )
    info_out.side_looking = translate_side_looking_from_model(info.side_looking)
    info_out.sensor_name = info.sensor_name
    info_out.description = info.description.value
    info_out.image_type = info.image_type.value
    info_out.projection = info.projection.value
    info_out.acquisition_station = info.acquisition_station.value
    info_out.processing_center = info.processing_center.value
    info_out.processing_software = info.processing_software.value
    if info.external_calibration_factor is not None:
        info_out.external_calibration_factor = info.external_calibration_factor
    if info.data_take_id is not None:
        info_out.data_take_id = info.data_take_id

    return info_out


def translate_dataset_info_to_model(
    info: metadata.DataSetInfo,
) -> metadata_models.DataSetInfoType:
    """Translate dataset info to model"""
    if info.description is None:
        raise RuntimeError(
            "Description field in DataSetInfo is required: cannot be 'None'"
        )
    if info.acquisition_mode is None:
        raise RuntimeError(
            "AcquisitionMode field in DataSetInfo is required: cannot be 'None'"
        )
    if info.image_type is None:
        raise RuntimeError(
            "ImageType field in DataSetInfo is required: cannot be 'None'"
        )
    if info.projection is None:
        raise RuntimeError(
            "Projection field in DataSetInfo is required: cannot be 'None'"
        )
    if info.acquisition_station is None:
        raise RuntimeError(
            "AcquisitionStation field in DataSetInfo is required: cannot be 'None'"
        )
    if info.acquisition_station is None:
        raise RuntimeError(
            "AcquisitionStation field in DataSetInfo is required: cannot be 'None'"
        )
    if info.processing_center is None:
        raise RuntimeError(
            "ProcessingCenter field in DataSetInfo is required: cannot be 'None'"
        )
    if info.processing_software is None:
        raise RuntimeError(
            "ProcessingCenter field in DataSetInfo is required: cannot be 'None'"
        )
    if info.side_looking is None:
        raise RuntimeError(
            "SideLooking field in DataSetInfo is required: cannot be 'None'"
        )

    def _optional_pdt_to_type(date: Optional[PreciseDateTime], output_type):
        if date is None:
            return output_type("NOT_AVAILABLE")
        return output_type(str(date))

    return metadata_models.DataSetInfoType(
        sensor_name=info.sensor_name,
        description=metadata_models.DataSetInfoType.Description(info.description),
        sense_date=_optional_pdt_to_type(
            info.sense_date, metadata_models.DataSetInfoType.SenseDate
        ),
        acquisition_mode=metadata_models.DataSetInfoType.AcquisitionMode(
            info.acquisition_mode
        ),
        image_type=metadata_models.DataSetInfoType.ImageType(info.image_type),
        projection=metadata_models.DataSetInfoType.Projection(info.projection),
        acquisition_station=metadata_models.DataSetInfoType.AcquisitionStation(
            info.acquisition_station
        ),
        processing_center=metadata_models.DataSetInfoType.ProcessingCenter(
            info.processing_center
        ),
        processing_date=_optional_pdt_to_type(
            info.processing_date, metadata_models.DataSetInfoType.ProcessingDate
        ),
        processing_software=metadata_models.DataSetInfoType.ProcessingSoftware(
            info.processing_software
        ),
        fc_hz=metadata_models.DataSetInfoType.FcHz(info.fc_hz),
        side_looking=translate_side_looking_to_model(info.side_looking),
        external_calibration_factor=info.external_calibration_factor,
        data_take_id=info.data_take_id,
    )


def translate_geo_point_from_model(
    point: metadata_models.PointType,
) -> metadata.GeoPoint:
    """Translate geo point from model"""
    assert len(point.val) == 5

    lat = point.val[0].value
    lon = point.val[1].value
    height = point.val[2].value
    theta_inc = point.val[3].value
    theta_look = point.val[4].value
    assert lat is not None and lon is not None and height is not None
    assert theta_inc is not None and theta_look is not None
    return metadata.GeoPoint(lat, lon, height, theta_inc, theta_look)


def translate_geo_point_to_model(point: metadata.GeoPoint) -> metadata_models.PointType:
    """Translate geo point to model"""
    return metadata_models.PointType(
        [
            metadata_models.PointType.Val(point.lat),
            metadata_models.PointType.Val(point.lon),
            metadata_models.PointType.Val(point.height),
            metadata_models.PointType.Val(point.theta_inc),
            metadata_models.PointType.Val(point.theta_look),
        ]
    )


def translate_ground_corner_points_from_model(
    corners: metadata_models.GroundCornersPointsType,
) -> metadata.GroundCornerPoints:
    """Translate ground corner points from model"""
    assert (
        corners.easting_grid_size is not None
        and corners.easting_grid_size.value is not None
    )
    assert (
        corners.northing_grid_size is not None
        and corners.northing_grid_size.value is not None
    )
    assert corners.north_west is not None and corners.north_west.point is not None
    assert corners.north_east is not None and corners.north_east.point is not None
    assert corners.south_west is not None and corners.south_west.point is not None
    assert corners.south_east is not None and corners.south_east.point is not None
    assert corners.center is not None and corners.center.point is not None
    corners_metadata = metadata.GroundCornerPoints()
    corners_metadata.easting_grid_size = corners.easting_grid_size.value
    corners_metadata.northing_grid_size = corners.northing_grid_size.value
    corners_metadata.nw_point = translate_geo_point_from_model(corners.north_west.point)
    corners_metadata.ne_point = translate_geo_point_from_model(corners.north_east.point)
    corners_metadata.sw_point = translate_geo_point_from_model(corners.south_west.point)
    corners_metadata.se_point = translate_geo_point_from_model(corners.south_east.point)
    corners_metadata.center_point = translate_geo_point_from_model(corners.center.point)
    return corners_metadata


def translate_ground_corner_points_to_model(
    corners: metadata.GroundCornerPoints,
) -> metadata_models.GroundCornersPointsType:
    """Translate ground corner points"""
    return metadata_models.GroundCornersPointsType(
        easting_grid_size=metadata_models.GroundCornersPointsType.EastingGridSize(
            corners.easting_grid_size
        ),
        northing_grid_size=metadata_models.GroundCornersPointsType.NorthingGridSize(
            corners.northing_grid_size
        ),
        north_west=metadata_models.GroundCornersPointsType.NorthWest(
            translate_geo_point_to_model(corners.nw_point)
        ),
        north_east=metadata_models.GroundCornersPointsType.NorthEast(
            translate_geo_point_to_model(corners.ne_point)
        ),
        south_west=metadata_models.GroundCornersPointsType.SouthWest(
            translate_geo_point_to_model(corners.sw_point)
        ),
        south_east=metadata_models.GroundCornersPointsType.SouthEast(
            translate_geo_point_to_model(corners.se_point)
        ),
        center=metadata_models.GroundCornersPointsType.Center(
            translate_geo_point_to_model(corners.center_point)
        ),
    )


def translate_swath_info_from_model(
    info: metadata_models.SwathInfoType,
) -> metadata.SwathInfo:
    """Translate swath info from model"""
    assert info.swath is not None
    assert info.polarization is not None
    assert (
        info.swath_acquisition_order is not None
        and info.swath_acquisition_order.value is not None
    )
    assert info.rank is not None and info.rank.value is not None
    assert info.range_delay_bias is not None and info.range_delay_bias.value is not None
    assert info.acquisition_start_time is not None
    assert (
        info.azimuth_steering_rate_reference_time is not None
        and info.azimuth_steering_rate_reference_time.value is not None
    )
    assert (
        info.azimuth_steering_rate_pol is not None
        and len(info.azimuth_steering_rate_pol.val) == 3
    )
    assert info.acquisition_prf is not None
    assert info.echoes_per_burst is not None

    swath_info_metadata = metadata.SwathInfo(
        swath_i=info.swath.value,
        polarization_i=translate_polarization_from_model(info.polarization),
        acquisition_prf_i=info.acquisition_prf,
    )
    swath_info_metadata.swath_acquisition_order = info.swath_acquisition_order.value
    swath_info_metadata.rank = info.rank.value
    swath_info_metadata.range_delay_bias = info.range_delay_bias.value
    swath_info_metadata.acquisition_start_time = PreciseDateTime.from_utc_string(
        info.acquisition_start_time.value
    )
    swath_info_metadata.azimuth_steering_rate_reference_time = (
        info.azimuth_steering_rate_reference_time.value
    )
    swath_info_metadata.azimuth_steering_rate_pol = tuple(
        coeff.value for coeff in info.azimuth_steering_rate_pol.val
    )
    swath_info_metadata.echoes_per_burst = info.echoes_per_burst
    swath_info_metadata.channel_delay = info.channel_delay
    swath_info_metadata.rx_gain = info.rx_gain

    return swath_info_metadata


def translate_swath_info_to_model(
    info: metadata.SwathInfo,
) -> metadata_models.SwathInfoType:
    """Translate swath info to model"""

    if info.swath is None:
        raise RuntimeError("Swath field in SwathInfo is required: cannot be 'None'")

    return metadata_models.SwathInfoType(
        swath=metadata_models.SwathInfoType.Swath(info.swath),
        swath_acquisition_order=metadata_models.SwathInfoType.SwathAcquisitionOrder(
            info.swath_acquisition_order
        ),
        polarization=translate_polarization_to_model(info.polarization),
        rank=metadata_models.SwathInfoType.Rank(info.rank),
        range_delay_bias=metadata_models.SwathInfoType.RangeDelayBias(
            info.range_delay_bias, translate_unit_to_model(info.range_delay_bias_unit)
        ),
        acquisition_start_time=metadata_models.SwathInfoType.AcquisitionStartTime(
            str(info.acquisition_start_time),
            translate_unit_to_model(info.acquisition_start_time_unit),
        ),
        azimuth_steering_rate_reference_time=metadata_models.DoubleWithUnit(
            info.azimuth_steering_rate_reference_time,
            translate_unit_to_model(info.az_steering_rate_ref_time_unit),
        ),
        azimuth_steering_rate_pol=metadata_models.SwathInfoType.AzimuthSteeringRatePol(
            [
                metadata_models.SwathInfoType.AzimuthSteeringRatePol.Val(
                    info.azimuth_steering_rate_pol[0], n=1
                ),
                metadata_models.SwathInfoType.AzimuthSteeringRatePol.Val(
                    info.azimuth_steering_rate_pol[1], n=2
                ),
                metadata_models.SwathInfoType.AzimuthSteeringRatePol.Val(
                    info.azimuth_steering_rate_pol[2], n=3
                ),
            ]
        ),
        acquisition_prf=info.acquisition_prf,
        echoes_per_burst=info.echoes_per_burst,
        channel_delay=info.channel_delay,
        rx_gain=info.rx_gain,
    )


def translate_sampling_constants_from_model(
    constants: metadata_models.SamplingConstantsType,
) -> metadata.SamplingConstants:
    """Translate sampling constants from model"""
    assert constants.frg_hz is not None and constants.brg_hz is not None
    assert constants.faz_hz is not None and constants.baz_hz is not None
    return metadata.SamplingConstants(
        frg_hz_i=constants.frg_hz.value,
        brg_hz_i=constants.brg_hz.value,
        faz_hz_i=constants.faz_hz.value,
        baz_hz_i=constants.baz_hz.value,
    )


def translate_sampling_constants_to_model(
    constants: metadata.SamplingConstants,
) -> metadata_models.SamplingConstantsType:
    """Translate sampling constants to model"""
    return metadata_models.SamplingConstantsType(
        frg_hz=metadata_models.SamplingConstantsType.FrgHz(
            constants.frg_hz, unit=translate_unit_to_model(constants.frg_hz_unit)
        ),
        brg_hz=metadata_models.SamplingConstantsType.BrgHz(
            constants.brg_hz, unit=translate_unit_to_model(constants.brg_hz_unit)
        ),
        faz_hz=metadata_models.SamplingConstantsType.FazHz(
            constants.faz_hz, unit=translate_unit_to_model(constants.faz_hz_unit)
        ),
        baz_hz=metadata_models.SamplingConstantsType.BazHz(
            constants.baz_hz, unit=translate_unit_to_model(constants.baz_hz_unit)
        ),
    )


def translate_acquisition_time_line_from_model(
    time_line: metadata_models.AcquisitionTimelineType,
) -> metadata.AcquisitionTimeLine:
    """Translate acquisition time line from model"""
    assert time_line.missing_lines_number is not None
    assert time_line.missing_lines_azimuthtimes is not None
    assert time_line.swst_changes_number is not None
    assert time_line.swst_changes_azimuthtimes is not None
    assert time_line.swst_changes_values is not None
    assert time_line.noise_packets_number is not None
    assert time_line.noise_packets_azimuthtimes is not None
    assert time_line.internal_calibration_number is not None
    assert time_line.internal_calibration_azimuthtimes is not None

    def to_list(model_list: Any) -> Optional[List]:
        """Convert model list to list"""
        output = []
        for element in model_list.val:
            assert element.value is not None
            output.append(element.value)

        return output if output else None

    if time_line.swl_changes_number:
        swl_changes_number = time_line.swl_changes_number
        assert time_line.swl_changes_azimuthtimes is not None
        swl_changes_azimuth_times = to_list(time_line.swl_changes_azimuthtimes)
        swl_changes_values = to_list(time_line.swl_changes_values)
    else:
        swl_changes_number = 0
        swl_changes_azimuth_times = None
        swl_changes_values = None

    if time_line.prf_changes_number:
        prf_changes_number = time_line.prf_changes_number
        assert time_line.prf_changes_azimuthtimes is not None
        prf_changes_azimuth_times = to_list(time_line.prf_changes_azimuthtimes)
        prf_changes_values = to_list(time_line.prf_changes_values)
    else:
        prf_changes_number = 0
        prf_changes_azimuth_times = None
        prf_changes_values = None

    output_time_line = metadata.AcquisitionTimeLine(
        missing_lines_number_i=time_line.missing_lines_number,
        missing_lines_azimuth_times_i=to_list(time_line.missing_lines_azimuthtimes),
        swst_changes_number_i=time_line.swst_changes_number,
        swst_changes_azimuth_times_i=to_list(time_line.swst_changes_azimuthtimes),
        swst_changes_values_i=to_list(time_line.swst_changes_values),
        noise_packets_number_i=time_line.noise_packets_number,
        noise_packets_azimuth_times_i=to_list(time_line.noise_packets_azimuthtimes),
        internal_calibration_number_i=time_line.internal_calibration_number,
        internal_calibration_azimuth_times_i=to_list(
            time_line.internal_calibration_azimuthtimes
        ),
        swl_changes_number_i=swl_changes_number,
        swl_changes_azimuth_times_i=swl_changes_azimuth_times,
        swl_changes_values_i=swl_changes_values,
        prf_changes_number_i=prf_changes_number,
        prf_changes_azimuth_times_i=prf_changes_azimuth_times,
        prf_changes_values_i=prf_changes_values,
        chirp_period=time_line.chirp_period,
    )

    if time_line.duplicated_lines_number not in (None, 0):
        output_time_line.duplicated_lines = (
            time_line.duplicated_lines_number,
            to_list(time_line.duplicated_lines_azimuthtimes),
        )

    return output_time_line


def translate_acquisition_time_line_to_model(
    time_line: metadata.AcquisitionTimeLine,
) -> metadata_models.AcquisitionTimelineType:
    """Translate acquisition time line from model"""

    def to_model(values: Optional[List[float]], unit: str, output_type) -> Any:
        return output_type(
            [
                output_type.Val(
                    value,
                    unit=translate_unit_to_model(unit),
                )
                for value in values
            ]
            if values is not None
            else []
        )

    def to_model_one_list(
        number: int, values: Optional[List[float]], unit: str, output_element_type
    ) -> Tuple[int, Optional[Any]]:
        return number, to_model(values, unit, output_element_type)

    def to_model_two_list(
        number: int,
        values_one: Optional[List[float]],
        values_two: Optional[List[float]],
        unit_one: str,
        unit_two: str,
        type_one,
        type_two,
    ) -> Tuple[Optional[int], Optional[Any], Optional[Any]]:
        return (
            number,
            to_model(values_one, unit_one, type_one),
            to_model(values_two, unit_two, type_two),
        )

    missing_lines_number, missing_lines_azimuth_times = to_model_one_list(
        *time_line.missing_lines,
        time_line.missing_lines_azimuth_times_unit,
        metadata_models.AcquisitionTimelineType.MissingLinesAzimuthtimes,
    )

    (
        swst_changes_number,
        swst_changes_azimuth_times,
        swst_changes_values,
    ) = to_model_two_list(
        *time_line.swst_changes,
        time_line.swst_changes_azimuth_times_unit,
        time_line.swst_changes_values_unit,
        metadata_models.AcquisitionTimelineType.SwstChangesAzimuthtimes,
        metadata_models.AcquisitionTimelineType.SwstChangesValues,
    )

    noise_packets_number, noise_packets_azimuthtimes = to_model_one_list(
        *time_line.noise_packet,
        time_line.noise_packets_azimuth_times_unit,
        metadata_models.AcquisitionTimelineType.NoisePacketsAzimuthtimes,
    )

    internal_calibration_number, internal_calibration_azimuthtimes = to_model_one_list(
        *time_line.internal_calibration,
        time_line.internal_calibration_azimuth_times_unit,
        metadata_models.AcquisitionTimelineType.InternalCalibrationAzimuthtimes,
    )

    swl_changes_number, swl_changes_azimuth_times, swl_changes_values = (
        None,
        None,
        None,
    )
    if time_line.swl_changes[0] > 0:
        (
            swl_changes_number,
            swl_changes_azimuth_times,
            swl_changes_values,
        ) = to_model_two_list(
            *time_line.swl_changes,
            time_line.swl_changes_azimuth_times_unit,
            time_line.swl_changes_values_unit,
            metadata_models.AcquisitionTimelineType.SwlChangesAzimuthtimes,
            metadata_models.AcquisitionTimelineType.SwlChangesValues,
        )

    prf_changes_number, prf_changes_azimuth_times, prf_changes_values = (
        None,
        None,
        None,
    )
    if time_line.prf_changes[0] > 0:
        (
            prf_changes_number,
            prf_changes_azimuth_times,
            prf_changes_values,
        ) = to_model_two_list(
            *time_line.prf_changes,
            time_line.prf_changes_azimuth_times_unit,
            time_line.prf_changes_values_unit,
            metadata_models.AcquisitionTimelineType.PrfChangesAzimuthtimes,
            metadata_models.AcquisitionTimelineType.PrfChangesValues,
        )

    duplicated_lines_number, duplicated_lines_azimuthtimes = None, None
    if time_line.duplicated_lines[0] > 0:
        duplicated_lines_number, duplicated_lines_azimuthtimes = to_model_one_list(
            *time_line.duplicated_lines,
            time_line.duplicated_lines_azimuth_times_unit,
            metadata_models.AcquisitionTimelineType.DuplicatedLinesAzimuthtimes,
        )

    return metadata_models.AcquisitionTimelineType(
        missing_lines_number=missing_lines_number,
        missing_lines_azimuthtimes=missing_lines_azimuth_times,
        swst_changes_number=swst_changes_number,
        swst_changes_azimuthtimes=swst_changes_azimuth_times,
        swst_changes_values=swst_changes_values,
        noise_packets_number=noise_packets_number,
        noise_packets_azimuthtimes=noise_packets_azimuthtimes,
        internal_calibration_number=internal_calibration_number,
        internal_calibration_azimuthtimes=internal_calibration_azimuthtimes,
        swl_changes_number=swl_changes_number,
        swl_changes_azimuthtimes=swl_changes_azimuth_times,
        swl_changes_values=swl_changes_values,
        duplicated_lines_number=duplicated_lines_number,
        duplicated_lines_azimuthtimes=duplicated_lines_azimuthtimes,
        prf_changes_number=prf_changes_number,
        prf_changes_azimuthtimes=prf_changes_azimuth_times,
        prf_changes_values=prf_changes_values,
        chirp_period=time_line.chirp_period,
    )


def translate_attitude_from_model(
    attitude: metadata_models.AttitudeInfoType,
) -> metadata.AttitudeInfo:
    """Translate attitude from model"""
    assert attitude.yaw_deg is not None
    assert attitude.pitch_deg is not None
    assert attitude.roll_deg is not None
    assert attitude.t_ref_utc is not None
    assert attitude.dt_ypr_s is not None and attitude.dt_ypr_s.value is not None
    assert attitude.reference_frame is not None
    assert attitude.rotation_order is not None
    assert attitude.attitude_type is not None

    def to_list(angles: List) -> List[float]:
        angles_list = []
        for element in angles:
            assert element.value is not None
            angles_list.append(element.value)
        return angles_list

    attitude_metadata = metadata.AttitudeInfo(
        yaw=to_list(attitude.yaw_deg.val),
        pitch=to_list(attitude.pitch_deg.val),
        roll=to_list(attitude.roll_deg.val),
        t0=PreciseDateTime.from_utc_string(attitude.t_ref_utc),
        delta_t=attitude.dt_ypr_s.value,
        ref_frame=translate_reference_frame_from_model(attitude.reference_frame).value,
        rot_order=translate_rotation_order_from_model(attitude.rotation_order).value,
    )
    attitude_metadata.attitude_type = translate_attitude_type_from_model(
        attitude.attitude_type
    ).value
    return attitude_metadata


def translate_attitude_to_model(
    attitude: metadata.AttitudeInfo,
) -> metadata_models.AttitudeInfoType:
    """Translate attitude to model"""
    return metadata_models.AttitudeInfoType(
        t_ref_utc=str(attitude.reference_time),
        dt_ypr_s=metadata_models.AttitudeInfoType.DtYprS(attitude.time_step),
        n_ypr_n=metadata_models.AttitudeInfoType.NYprN(
            attitude.attitude_records_number
        ),
        yaw_deg=metadata_models.AttitudeInfoType.YawDeg(
            [
                metadata_models.AttitudeInfoType.YawDeg.Val(yaw, n=index + 1)
                for index, yaw in enumerate(attitude.yaw_vector)
            ]
        ),
        pitch_deg=metadata_models.AttitudeInfoType.PitchDeg(
            [
                metadata_models.AttitudeInfoType.PitchDeg.Val(pitch, n=index + 1)
                for index, pitch in enumerate(attitude.pitch_vector)
            ]
        ),
        roll_deg=metadata_models.AttitudeInfoType.RollDeg(
            [
                metadata_models.AttitudeInfoType.RollDeg.Val(roll, n=index + 1)
                for index, roll in enumerate(attitude.roll_vector)
            ]
        ),
        reference_frame=translate_reference_frame_to_model(attitude.reference_frame),
        rotation_order=translate_rotation_order_to_model(attitude.rotation_order),
        attitude_type=translate_attitude_type_to_model(attitude.attitude_type),
    )


def _fill_lines_per_burst_list(
    changes: List[metadata_models.BurstInfoType.LinesPerBurstChangeList.Lines],
    burst_number: int,
) -> List[int]:
    """Fill lines per burst list"""
    changes_dict: Dict[int, int] = {}
    for change in changes:
        assert change.from_burst is not None
        assert change.value is not None
        changes_dict[change.from_burst] = change.value

    lines_per_burst = []
    for burst_index in range(burst_number):
        for change_index in sorted(changes_dict.keys(), reverse=True):
            if burst_index + 1 >= change_index:
                lines_per_burst.append(changes_dict[change_index])
                break
    return lines_per_burst


def translate_burst_info_from_model(
    info: metadata_models.BurstInfoType,
) -> metadata.BurstInfo:
    """Translate burst info from model"""
    assert info.number_of_bursts is not None
    assert (
        info.burst_repetition_frequency is not None
        and info.burst_repetition_frequency.value is not None
    )

    output_info = metadata.BurstInfo(
        burst_repetition_frequency=info.burst_repetition_frequency.value
    )
    if info.lines_per_burst is not None:
        lines_per_burst = info.number_of_bursts * [info.lines_per_burst]
    else:
        assert info.lines_per_burst_change_list
        lines_per_burst = _fill_lines_per_burst_list(
            info.lines_per_burst_change_list.lines, info.number_of_bursts
        )

    assert len(info.burst) == info.number_of_bursts
    for index, (burst, lines) in enumerate(zip(info.burst, lines_per_burst)):
        burst: metadata_models.BurstType
        assert burst.range_start_time is not None
        assert burst.azimuth_start_time is not None
        assert burst.n is not None and burst.n == index + 1
        output_info.add_burst(
            range_start_time_i=burst.range_start_time.value,
            azimuth_start_time_i=PreciseDateTime.from_utc_string(
                burst.azimuth_start_time.value
            ),
            lines_i=lines,
            burst_center_azimuth_shift_i=(
                burst.burst_center_azimuth_shift.value
                if burst.burst_center_azimuth_shift is not None
                else None
            ),
        )

    return output_info


def translate_burst_info_to_model(
    info: metadata.BurstInfo,
) -> metadata_models.BurstInfoType:
    """Translate burst info to model"""
    changes: Dict[int, int] = {}
    last_num_lines = 0
    for index in range(info.get_number_of_bursts()):
        lines = info.get_burst(index).lines
        if lines != last_num_lines:
            changes[index] = last_num_lines = lines

    change_list = metadata_models.BurstInfoType.LinesPerBurstChangeList(
        [
            metadata_models.BurstInfoType.LinesPerBurstChangeList.Lines(
                lines, from_burst + 1
            )
            for from_burst, lines in changes.items()
        ]
    )

    burst_list = []
    for index in range(info.get_number_of_bursts()):
        burst = info.get_burst(index)
        burst_model = metadata_models.BurstType(
            range_start_time=metadata_models.DoubleWithUnit(
                burst.range_start_time, unit=metadata_models.Units.S
            ),
            azimuth_start_time=metadata_models.StringWithUnit(
                value=str(burst.azimuth_start_time), unit=metadata_models.Units.UTC
            ),
            burst_center_azimuth_shift=(
                metadata_models.DoubleWithUnit(
                    burst.burst_center_azimuth_shift, unit=metadata_models.Units.S
                )
                if burst.burst_center_azimuth_shift is not None
                else None
            ),
            n=index + 1,
        )
        burst_list.append(burst_model)

    return metadata_models.BurstInfoType(
        number_of_bursts=info.get_number_of_bursts(),
        burst_repetition_frequency=metadata_models.DoubleWithUnit(
            info.burst_repetition_frequency, metadata_models.Units.HZ
        ),
        lines_per_burst=None,  # Lines per burst is considered deprecated
        lines_per_burst_change_list=change_list,
        burst=burst_list,
    )


def translate_state_vectors_from_model(
    state_vectors: metadata_models.StateVectorDataType,
) -> metadata.StateVectors:
    """Translate state vectors from model"""
    assert state_vectors.orbit_number is not None
    assert state_vectors.track is not None
    assert state_vectors.orbit_direction is not None
    assert state_vectors.p_sv_m is not None
    assert state_vectors.v_sv_m_os is not None
    assert state_vectors.t_ref_utc is not None
    assert state_vectors.dt_sv_s is not None and state_vectors.dt_sv_s.value is not None
    assert state_vectors.n_sv_n is not None and state_vectors.n_sv_n.value is not None

    number_of_state_vectors = state_vectors.n_sv_n.value

    positions = np.zeros((number_of_state_vectors, 3))
    velocities = np.zeros((number_of_state_vectors, 3))

    for index, (pos, vel) in enumerate(
        zip(state_vectors.p_sv_m.val, state_vectors.v_sv_m_os.val)
    ):
        assert pos is not None and vel is not None
        assert pos.value is not None and vel.value is not None
        assert pos.n == index + 1 and vel.n == index + 1
        state_vector_index = index // 3
        component_index = index % 3

        positions[state_vector_index, component_index] = pos.value
        velocities[state_vector_index, component_index] = vel.value

    state_vectors_metadata = metadata.StateVectors(
        positions,
        velocities,
        PreciseDateTime.from_utc_string(state_vectors.t_ref_utc),
        state_vectors.dt_sv_s.value,
    )

    if state_vectors.orbit_number != "NOT_AVAILABLE":
        state_vectors_metadata.orbit_number = int(state_vectors.orbit_number)

    if state_vectors.track != "NOT_AVAILABLE":
        state_vectors_metadata.track_number = int(state_vectors.track)

    anx_position = None
    if state_vectors.ascending_node_coords is not None:
        anx_position = [
            element.value for element in state_vectors.ascending_node_coords.val
        ]

    anx_time = None
    if state_vectors.ascending_node_time is not None:
        anx_time = PreciseDateTime.from_utc_string(state_vectors.ascending_node_time)

    state_vectors_metadata.set_anx_info(anx_time, anx_position)

    return state_vectors_metadata


def translate_state_vectors_to_model(
    state_vectors: metadata.StateVectors,
) -> metadata_models.StateVectorDataType:
    """Translate state vectors to model"""

    position = metadata_models.StateVectorDataType.PSvM([])
    velocity = metadata_models.StateVectorDataType.VSvMOs([])
    for index, (pos, vel) in enumerate(
        zip(state_vectors.position_vector, state_vectors.velocity_vector)
    ):
        for component, (pos_comp, vel_comp) in enumerate(zip(pos, vel)):
            index_current = index * 3 + component + 1

            position.val.append(
                metadata_models.StateVectorDataType.PSvM.Val(pos_comp, n=index_current)
            )
            velocity.val.append(
                metadata_models.StateVectorDataType.VSvMOs.Val(
                    vel_comp, n=index_current
                )
            )

    state_vectors_model = metadata_models.StateVectorDataType(
        p_sv_m=position,
        v_sv_m_os=velocity,
        orbit_number=(
            "NOT_AVAILABLE"
            if state_vectors.orbit_number < 0
            else str(state_vectors.orbit_number)
        ),
        track=(
            "NOT_AVAILABLE"
            if state_vectors.track_number < 0
            else str(state_vectors.track_number)
        ),
        orbit_direction=translate_orbit_direction_to_model(
            state_vectors.orbit_direction
        ),
        t_ref_utc=str(state_vectors.reference_time),
        dt_sv_s=metadata_models.StateVectorDataType.DtSvS(
            state_vectors.time_step, unit=metadata_models.Units.S
        ),
        n_sv_n=metadata_models.StateVectorDataType.NSvN(
            state_vectors.number_of_state_vectors
        ),
    )

    if state_vectors.anx_position is not None:
        state_vectors_model.ascending_node_coords = (
            metadata_models.StateVectorDataType.AscendingNodeCoords(
                [
                    metadata_models.StateVectorDataType.AscendingNodeCoords.Val(
                        state_vectors.anx_position[0]
                    ),
                    metadata_models.StateVectorDataType.AscendingNodeCoords.Val(
                        state_vectors.anx_position[1]
                    ),
                    metadata_models.StateVectorDataType.AscendingNodeCoords.Val(
                        state_vectors.anx_position[2]
                    ),
                ]
            )
        )

    if state_vectors.anx_time is not None:
        state_vectors_model.ascending_node_time = str(state_vectors.anx_time)

    return state_vectors_model


def translate_polynomial_from_model(
    poly: metadata_models.PolyType, specific_type=metadata._Poly2D
) -> metadata._Poly2D:
    """Translate polynomial from model"""
    assert poly.pol is not None
    assert poly.trg0_s is not None
    assert poly.taz0_utc is not None

    return specific_type(
        i_ref_az=PreciseDateTime.from_utc_string(poly.taz0_utc.value),
        i_ref_rg=poly.trg0_s.value,
        i_coefficients=[elem.value for elem in poly.pol.val],
    )


def translate_polynomial_to_model(poly: metadata._Poly2D) -> metadata_models.PolyType:
    """Translate polynomial to model"""
    if poly.coefficients is None:
        raise RuntimeError(
            "Coefficients are required in 2D polynomial: cannot be 'None'"
        )

    return metadata_models.PolyType(
        pol=metadata_models.PolyType.Pol(
            [
                metadata_models.PolyType.Pol.Val(
                    coeff, unit=translate_unit_to_model(unit), n=index + 1
                )
                for index, (coeff, unit) in enumerate(
                    zip(poly.coefficients, poly.get_units())
                )
            ]
        ),
        trg0_s=metadata_models.PolyType.Trg0S(
            poly.t_ref_rg, unit=metadata_models.Units.S
        ),
        taz0_utc=metadata_models.PolyType.Taz0Utc(
            str(poly.t_ref_az), unit=metadata_models.Units.UTC
        ),
    )


def translate_polynomial_list_from_model(
    poly_list: List[metadata_models.PolyType],
    specific_type=metadata._Poly2DVector,
) -> metadata._Poly2DVector:
    """Translate polynomial list from model"""
    return specific_type(
        [
            translate_polynomial_from_model(poly, specific_type._SINGLE_POLY_TYPE)
            for poly in poly_list
        ]
    )


def translate_polynomial_list_to_model(
    poly_list: metadata._Poly2DVector,
) -> List[metadata_models.PolyType]:
    """Translate polynomial list from model"""

    def _add_number_and_total(poly: metadata_models.PolyType, number: int, total: int):
        poly.number = number
        poly.total = total
        return poly

    return [
        _add_number_and_total(
            translate_polynomial_to_model(poly), index + 1, len(poly_list)
        )
        for index, poly in enumerate(poly_list)
    ]


def translate_coreg_polynomial_from_model(
    poly: metadata_models.PolyCoregType,
) -> metadata.CoregPoly:
    """Translate coregistration polynomial from model"""
    assert poly.pol_rg is not None
    assert poly.pol_az is not None
    assert poly.trg0_s is not None
    assert poly.taz0_utc is not None

    return metadata.CoregPoly(
        i_ref_az=PreciseDateTime.from_utc_string(poly.taz0_utc.value),
        i_ref_rg=poly.trg0_s.value,
        i_coefficients_az=[elem.value for elem in poly.pol_az.val],
        i_coefficients_rg=[elem.value for elem in poly.pol_rg.val],
    )


def translate_coreg_polynomial_to_model(
    poly: metadata.CoregPoly,
) -> metadata_models.PolyCoregType:
    """Translate coregistration polynomial to model"""
    if poly.azimuth_poly.coefficients is None:
        raise RuntimeError(
            "Azimuth Coefficients are required in Coreg Polynomial: cannot be 'None'"
        )

    if poly.range_poly.coefficients is None:
        raise RuntimeError(
            "Range Coefficients are required in Coreg Polynomial: cannot be 'None'"
        )

    return metadata_models.PolyCoregType(
        pol_az=metadata_models.PolyCoregType.PolAz(
            [
                metadata_models.PolyCoregType.PolAz.Val(coeff, n=index + 1)
                for index, coeff in enumerate(poly.azimuth_poly.coefficients)
            ]
        ),
        pol_rg=metadata_models.PolyCoregType.PolRg(
            [
                metadata_models.PolyCoregType.PolRg.Val(coeff, n=index + 1)
                for index, coeff in enumerate(poly.range_poly.coefficients)
            ]
        ),
        trg0_s=metadata_models.PolyCoregType.Trg0S(
            poly.ref_range_time, unit=metadata_models.Units.S
        ),
        taz0_utc=metadata_models.PolyCoregType.Taz0Utc(
            str(poly.ref_azimuth_time), unit=metadata_models.Units.UTC
        ),
    )


def translate_coreg_polynomial_list_from_model(
    poly_list: List[metadata_models.PolyCoregType],
) -> metadata.CoregPolyVector:
    """Translate coregistration polynomial list from model"""
    return metadata.CoregPolyVector(
        [translate_coreg_polynomial_from_model(poly) for poly in poly_list]
    )


def translate_coreg_polynomial_list_to_model(
    poly_list: metadata.CoregPolyVector,
) -> List[metadata_models.PolyCoregType]:
    """Translate coregistration polynomial list to model"""

    def _add_number_and_total(
        poly: metadata_models.PolyCoregType, number: int, total: int
    ):
        poly.number = number
        poly.total = total
        return poly

    return [
        _add_number_and_total(
            translate_coreg_polynomial_to_model(poly),
            number=index + 1,
            total=len(poly_list),
        )
        for index, poly in enumerate(poly_list)
    ]


def translate_data_statistics_from_model(
    stat: metadata_models.DataStatisticsType,
) -> metadata.DataStatistics:
    """Translate data statistics from model"""
    assert stat.num_samples is not None and stat.num_samples.value is not None
    assert stat.max_i is not None and stat.max_i.value is not None
    assert stat.min_i is not None and stat.min_i.value is not None
    assert stat.max_q is not None and stat.max_q.value is not None
    assert stat.min_q is not None and stat.min_q.value is not None
    assert stat.sum_i is not None and stat.sum_i.value is not None
    assert stat.sum_q is not None and stat.sum_q.value is not None
    assert stat.sum2_i is not None and stat.sum2_i.value is not None
    assert stat.sum2_q is not None and stat.sum2_q.value is not None
    assert stat.std_dev_i is not None and stat.std_dev_i.value is not None
    assert stat.std_dev_q is not None and stat.std_dev_q.value is not None

    stat_metadata = metadata.DataStatistics(
        stat.num_samples.value,
        stat.max_i.value,
        stat.max_q.value,
        stat.min_i.value,
        stat.min_q.value,
        stat.sum_i.value,
        stat.sum_q.value,
        stat.sum2_i.value,
        stat.sum2_q.value,
        stat.std_dev_i.value,
        stat.std_dev_q.value,
    )

    if stat.statistics_list is not None:
        for block_stat in stat.statistics_list.data_block_statistic:
            assert block_stat.line_start is not None
            assert block_stat.line_stop is not None
            assert (
                block_stat.num_samples is not None
                and block_stat.num_samples.value is not None
            )
            assert block_stat.max_i is not None and block_stat.max_i.value is not None
            assert block_stat.max_q is not None and block_stat.max_q.value is not None
            assert block_stat.min_i is not None and block_stat.min_i.value is not None
            assert block_stat.min_q is not None and block_stat.min_q.value is not None
            assert block_stat.sum_i is not None and block_stat.sum_i.value is not None
            assert block_stat.sum_q is not None and block_stat.sum_q.value is not None
            assert block_stat.sum2_i is not None and block_stat.sum2_i.value is not None
            assert block_stat.sum2_q is not None and block_stat.sum2_q.value is not None

            stat_metadata.add_data_block_statistic(
                metadata.DataBlockStatistic(
                    block_stat.line_start,
                    block_stat.line_stop,
                    block_stat.num_samples.value,
                    block_stat.max_i.value,
                    block_stat.max_q.value,
                    block_stat.min_i.value,
                    block_stat.min_q.value,
                    block_stat.sum_i.value,
                    block_stat.sum_q.value,
                    block_stat.sum2_i.value,
                    block_stat.sum2_q.value,
                )
            )

    return stat_metadata


def translate_data_statistics_to_model(
    stat: metadata.DataStatistics,
) -> metadata_models.DataStatisticsType:
    """Translate data statistics to model"""

    stat_model = metadata_models.DataStatisticsType(
        num_samples=metadata_models.DataStatisticsType.NumSamples(stat.num_samples),
        max_i=metadata_models.DataStatisticsType.MaxI(stat.max_i),
        min_i=metadata_models.DataStatisticsType.MinI(stat.min_i),
        max_q=metadata_models.DataStatisticsType.MaxQ(stat.max_q),
        min_q=metadata_models.DataStatisticsType.MinQ(stat.min_q),
        sum_i=metadata_models.DataStatisticsType.SumI(stat.sum_i),
        sum_q=metadata_models.DataStatisticsType.SumQ(stat.sum_q),
        sum2_i=metadata_models.DataStatisticsType.Sum2I(stat.sum_2_i),
        sum2_q=metadata_models.DataStatisticsType.Sum2Q(stat.sum_2_q),
        std_dev_i=metadata_models.DataStatisticsType.StdDevI(stat.std_dev_i),
        std_dev_q=metadata_models.DataStatisticsType.StdDevQ(stat.std_dev_q),
    )
    if stat.get_number_of_data_block_statistic() != 0:
        stat_model.statistics_list = metadata_models.DataStatisticsType.StatisticsList(
            []
        )
        for index in range(stat.get_number_of_data_block_statistic()):
            block = stat.get_data_block_statistic(index)
            stat_model.statistics_list.data_block_statistic.append(
                metadata_models.DataBlockStatisticsType(
                    num_samples=metadata_models.DataBlockStatisticsType.NumSamples(
                        block.num_samples
                    ),
                    max_i=metadata_models.DataBlockStatisticsType.MaxI(block.max_i),
                    min_i=metadata_models.DataBlockStatisticsType.MinI(block.min_i),
                    max_q=metadata_models.DataBlockStatisticsType.MaxQ(block.max_q),
                    min_q=metadata_models.DataBlockStatisticsType.MinQ(block.min_q),
                    sum_i=metadata_models.DataBlockStatisticsType.SumI(block.sum_i),
                    sum_q=metadata_models.DataBlockStatisticsType.SumQ(block.sum_q),
                    sum2_i=metadata_models.DataBlockStatisticsType.Sum2I(block.sum_2_i),
                    sum2_q=metadata_models.DataBlockStatisticsType.Sum2Q(block.sum_2_q),
                )
            )

    return stat_model


def translate_sensor_names_from_model(name: metadata_models.SensorNamesType) -> str:
    """Translate sensor name from model"""
    return name.value


def translate_sensor_names_to_model(name: str) -> metadata_models.SensorNamesType:
    """Translate sensor name to model"""
    return metadata_models.SensorNamesType(name)


def translate_antenna_info_from_model(
    info: metadata_models.AntennaInfoType,
) -> metadata.AntennaInfo:
    """Translate antenna info from model"""
    assert info.sensor_name is not None
    assert info.acquisition_mode is not None
    assert info.beam_name is not None
    assert info.polarization is not None

    info_metadata = metadata.AntennaInfo(
        i_acquisition_beam=info.beam_name,
        i_polarization=translate_polarization_from_model(info.polarization),
        i_acquisition_mode=info.acquisition_mode.value,
        i_sensor_name=translate_sensor_names_from_model(info.sensor_name),
    )

    if info.lines_per_pattern is not None:
        info_metadata.lines_per_pattern = info.lines_per_pattern

    return info_metadata


def translate_antenna_info_to_model(
    info: metadata.AntennaInfo,
) -> metadata_models.AntennaInfoType:
    """Translate antenna info to model"""

    info_model = metadata_models.AntennaInfoType(
        beam_name=info.acquisition_beam,
        sensor_name=metadata_models.SensorNamesType(
            info.sensor_name if info.sensor_name is not None else "NOT SET"
        ),
        acquisition_mode=metadata_models.AcquisitionModeType(info.acquisition_mode),
        polarization=translate_polarization_to_model(info.polarization),
    )

    if info.lines_per_pattern != 0:
        info_model.lines_per_pattern = info.lines_per_pattern

    return info_model


def translate_pulse_direction_from_model(
    direction: metadata_models.PulseTypeDirection,
) -> metadata.EPulseDirection:
    """Translate pulse direction from model"""
    return metadata.EPulseDirection(direction.value)


def translate_pulse_direction_to_model(
    direction: metadata.EPulseDirection,
) -> metadata_models.PulseTypeDirection:
    """Translate pulse direction to model"""
    return metadata_models.PulseTypeDirection(direction.value)


def translate_pulse_from_model(pulse: metadata_models.PulseType) -> metadata.Pulse:
    """Translate pulse from model"""
    assert pulse.pulse_length is not None
    assert pulse.bandwidth is not None
    assert pulse.pulse_sampling_rate is not None

    return metadata.Pulse(
        i_pulse_length=pulse.pulse_length.value,
        i_bandwidth=pulse.bandwidth.value,
        i_pulse_sampling_rate=pulse.pulse_sampling_rate.value,
        i_pulse_energy=(
            pulse.pulse_energy.value if pulse.pulse_energy is not None else None
        ),
        i_pulse_start_frequency=(
            pulse.pulse_start_frequency.value
            if pulse.pulse_start_frequency is not None
            else None
        ),
        i_pulse_start_phase=(
            pulse.pulse_start_phase.value
            if pulse.pulse_start_phase is not None
            else None
        ),
        i_pulse_direction=(
            translate_pulse_direction_from_model(pulse.direction)
            if pulse.direction is not None
            else None
        ),
    )


def translate_pulse_to_model(pulse: metadata.Pulse) -> metadata_models.PulseType:
    """Translate pulse to model"""

    if pulse.bandwidth is None:
        raise RuntimeError("Bandwidth field in Pulse is required: cannot be 'None'")

    if pulse.pulse_sampling_rate is None:
        raise RuntimeError("Sampling Rate field in Pulse is required: cannot be 'None'")

    if pulse.pulse_length is None:
        raise RuntimeError("Pulse Length field in Pulse is required: cannot be 'None'")

    return metadata_models.PulseType(
        pulse_length=metadata_models.DoubleWithUnit(
            pulse.pulse_length, translate_unit_to_model(pulse.pulse_length_unit)
        ),
        bandwidth=metadata_models.DoubleWithUnit(
            pulse.bandwidth, translate_unit_to_model(pulse.bandwidth_unit)
        ),
        pulse_energy=(
            metadata_models.DoubleWithUnit(
                pulse.pulse_energy, translate_unit_to_model(pulse.pulse_energy_unit)
            )
            if pulse.pulse_energy is not None
            else None
        ),
        pulse_sampling_rate=metadata_models.DoubleWithUnit(
            pulse.pulse_sampling_rate,
            translate_unit_to_model(pulse.pulse_sampling_rate_unit),
        ),
        pulse_start_frequency=(
            metadata_models.DoubleWithUnit(
                pulse.pulse_start_frequency,
                translate_unit_to_model(pulse.pulse_start_frequency_unit),
            )
            if pulse.pulse_start_frequency is not None
            else None
        ),
        pulse_start_phase=(
            metadata_models.DoubleWithUnit(
                pulse.pulse_start_phase,
                translate_unit_to_model(pulse.pulse_start_phase_unit),
            )
            if pulse.pulse_start_phase is not None
            else None
        ),
        direction=(
            translate_pulse_direction_to_model(pulse.pulse_direction)
            if pulse.pulse_direction is not None
            else None
        ),
    )


def translate_metadata_from_model(
    model: metadata_models.AresysXmlDoc,
) -> metadata.MetaData:
    """Translate metadata model into corresponding object

    Parameters
    ----------
    model : metadata_models.AresysXmlDoc
        xsdata model

    Returns
    -------
    metadata.MetaData
        arepytools metadata
    """
    assert model.description is not None
    output_metadata = metadata.MetaData(description=model.description)

    for channel in model.channel:
        mdc = metadata.MetaDataChannel()

        mdc.contentID = channel.content_id
        mdc.number = channel.number
        mdc.total = channel.total

        if channel.raster_info is not None:
            mdc.insert_element(translate_raster_info_from_model(channel.raster_info))

        if channel.data_set_info is not None:
            mdc.insert_element(translate_dataset_info_from_model(channel.data_set_info))

        if channel.swath_info is not None:
            mdc.insert_element(translate_swath_info_from_model(channel.swath_info))

        if channel.sampling_constants is not None:
            mdc.insert_element(
                translate_sampling_constants_from_model(channel.sampling_constants)
            )

        if channel.acquisition_time_line is not None:
            mdc.insert_element(
                translate_acquisition_time_line_from_model(
                    channel.acquisition_time_line
                )
            )

        if channel.data_statistics is not None:
            mdc.insert_element(
                translate_data_statistics_from_model(channel.data_statistics)
            )

        if channel.burst_info is not None:
            mdc.insert_element(translate_burst_info_from_model(channel.burst_info))

        if channel.state_vector_data is not None:
            mdc.insert_element(
                translate_state_vectors_from_model(channel.state_vector_data)
            )

        if channel.doppler_centroid is not None:
            mdc.insert_element(
                translate_polynomial_list_from_model(
                    channel.doppler_centroid,
                    specific_type=metadata.DopplerCentroidVector,
                )
            )

        if channel.doppler_rate is not None:
            mdc.insert_element(
                translate_polynomial_list_from_model(
                    channel.doppler_rate,
                    specific_type=metadata.DopplerRateVector,
                )
            )

        if channel.tops_azimuth_modulation_rate is not None:
            mdc.insert_element(
                translate_polynomial_list_from_model(
                    channel.tops_azimuth_modulation_rate,
                    specific_type=metadata.TopsAzimuthModulationRateVector,
                )
            )

        if channel.slant_to_ground is not None:
            mdc.insert_element(
                translate_polynomial_list_from_model(
                    channel.slant_to_ground,
                    specific_type=metadata.SlantToGroundVector,
                )
            )

        if channel.ground_to_slant is not None:
            mdc.insert_element(
                translate_polynomial_list_from_model(
                    channel.ground_to_slant,
                    specific_type=metadata.GroundToSlantVector,
                )
            )

        if channel.slant_to_incidence is not None:
            mdc.insert_element(
                translate_polynomial_list_from_model(
                    channel.slant_to_incidence,
                    specific_type=metadata.SlantToIncidenceVector,
                )
            )

        if channel.slant_to_elevation is not None:
            mdc.insert_element(
                translate_polynomial_list_from_model(
                    channel.slant_to_elevation,
                    specific_type=metadata.SlantToElevationVector,
                )
            )

        if channel.attitude_info is not None:
            mdc.insert_element(translate_attitude_from_model(channel.attitude_info))

        if channel.ground_corner_points is not None:
            mdc.insert_element(
                translate_ground_corner_points_from_model(channel.ground_corner_points)
            )

        if channel.pulse is not None:
            mdc.insert_element(translate_pulse_from_model(channel.pulse))

        if channel.coreg_poly is not None:
            mdc.insert_element(
                translate_coreg_polynomial_list_from_model(channel.coreg_poly)
            )

        if channel.antenna_info is not None:
            mdc.insert_element(translate_antenna_info_from_model(channel.antenna_info))

        output_metadata.append_channel(mdc)

    return output_metadata


def translate_metadata_to_model(
    metadata_obj: metadata.MetaData,
) -> metadata_models.AresysXmlDoc:
    """Translate metadata model into corresponding object

    Parameters
    ----------
    model : metadata.MetaData
        arepytools metadata

    Returns
    -------
    metadata_models.AresysXmlDoc
        xsdata model
    """
    version = 2.1
    metadata_model = metadata_models.AresysXmlDoc(
        metadata_obj.get_number_of_channels(), version, metadata_obj.description
    )

    for channel_index in range(metadata_obj.get_number_of_channels()):
        metadata_channel = metadata_obj.get_metadata_channels(channel_index)

        number = (
            metadata_channel.number
            if metadata_channel.number is not None
            else channel_index + 1
        )

        total = (
            metadata_channel.total
            if metadata_channel.total is not None
            else metadata_obj.get_number_of_channels()
        )

        channel_model = metadata_models.AresysXmlDoc.Channel(
            number=number,
            total=total,
            content_id=metadata_channel.contentID,
        )

        raster_info: Optional[metadata.RasterInfo] = metadata_channel.get_element(
            "RasterInfo"
        )  # type: ignore
        sampling_constants: Optional[
            metadata.SamplingConstants
        ] = metadata_channel.get_element(
            "SamplingConstants"
        )  # type: ignore
        pulse: Optional[metadata.Pulse] = metadata_channel.get_element("Pulse")  # type: ignore
        swath_info: Optional[metadata.SwathInfo] = metadata_channel.get_element(
            "SwathInfo"
        )  # type: ignore
        data_set_info: Optional[metadata.DataSetInfo] = metadata_channel.get_element(
            "DataSetInfo"
        )  # type: ignore
        state_vectors: Optional[metadata.StateVectors] = metadata_channel.get_element(
            "StateVectors"
        )  # type: ignore
        attitude_info: Optional[metadata.AttitudeInfo] = metadata_channel.get_element(
            "AttitudeInfo"
        )  # type: ignore
        acquisition_time_line: Optional[
            metadata.AcquisitionTimeLine
        ] = metadata_channel.get_element(
            "AcquisitionTimeLine"
        )  # type: ignore
        ground_corner_points: Optional[
            metadata.GroundCornerPoints
        ] = metadata_channel.get_element(
            "GroundCornerPoints"
        )  # type: ignore
        burst_info: Optional[metadata.BurstInfo] = metadata_channel.get_element(
            "BurstInfo"
        )  # type: ignore
        doppler_centroid_vector: Optional[
            metadata.DopplerCentroidVector
        ] = metadata_channel.get_element(
            "DopplerCentroidVector"
        )  # type: ignore
        doppler_rate_vector: Optional[
            metadata.DopplerRateVector
        ] = metadata_channel.get_element(
            "DopplerRateVector"
        )  # type: ignore
        tops_azimuth_modulation_rate_vector: Optional[
            metadata.TopsAzimuthModulationRateVector
        ] = metadata_channel.get_element(
            "TopsAzimuthModulationRateVector"
        )  # type: ignore
        slant_to_ground_vector: Optional[
            metadata.SlantToGroundVector
        ] = metadata_channel.get_element(
            "SlantToGroundVector"
        )  # type: ignore
        ground_to_slant_vector: Optional[
            metadata.GroundToSlantVector
        ] = metadata_channel.get_element(
            "GroundToSlantVector"
        )  # type: ignore
        slant_to_incidence_vector: Optional[
            metadata.SlantToIncidenceVector
        ] = metadata_channel.get_element(
            "SlantToIncidenceVector"
        )  # type: ignore
        slant_to_elevation_vector: Optional[
            metadata.SlantToElevationVector
        ] = metadata_channel.get_element(
            "SlantToElevationVector"
        )  # type: ignore
        antenna_info: Optional[metadata.AntennaInfo] = metadata_channel.get_element(
            "AntennaInfo"
        )  # type: ignore
        data_statistics: Optional[
            metadata.DataStatistics
        ] = metadata_channel.get_element(
            "DataStatistics"
        )  # type: ignore
        coreg_poly_vector: Optional[
            metadata.CoregPolyVector
        ] = metadata_channel.get_element(
            "CoregPolyVector"
        )  # type: ignore

        if raster_info:
            channel_model.raster_info = translate_raster_info_to_model(raster_info)
        if sampling_constants:
            channel_model.sampling_constants = translate_sampling_constants_to_model(
                sampling_constants
            )
        if pulse:
            channel_model.pulse = translate_pulse_to_model(pulse)
        if swath_info:
            channel_model.swath_info = translate_swath_info_to_model(swath_info)
        if data_set_info:
            channel_model.data_set_info = translate_dataset_info_to_model(data_set_info)
        if state_vectors:
            channel_model.state_vector_data = translate_state_vectors_to_model(
                state_vectors
            )
        if attitude_info:
            channel_model.attitude_info = translate_attitude_to_model(attitude_info)
        if acquisition_time_line:
            channel_model.acquisition_time_line = (
                translate_acquisition_time_line_to_model(acquisition_time_line)
            )
        if ground_corner_points:
            channel_model.ground_corner_points = (
                translate_ground_corner_points_to_model(ground_corner_points)
            )
        if burst_info:
            channel_model.burst_info = translate_burst_info_to_model(burst_info)
        if doppler_centroid_vector:
            channel_model.doppler_centroid = translate_polynomial_list_to_model(
                doppler_centroid_vector
            )
        if doppler_rate_vector:
            channel_model.doppler_rate = translate_polynomial_list_to_model(
                doppler_rate_vector
            )
        if tops_azimuth_modulation_rate_vector:
            channel_model.tops_azimuth_modulation_rate = (
                translate_polynomial_list_to_model(tops_azimuth_modulation_rate_vector)
            )
        if slant_to_ground_vector:
            channel_model.slant_to_ground = translate_polynomial_list_to_model(
                slant_to_ground_vector
            )
        if ground_to_slant_vector:
            channel_model.ground_to_slant = translate_polynomial_list_to_model(
                ground_to_slant_vector
            )
        if slant_to_elevation_vector:
            channel_model.slant_to_elevation = translate_polynomial_list_to_model(
                slant_to_elevation_vector
            )
        if slant_to_incidence_vector:
            channel_model.slant_to_incidence = translate_polynomial_list_to_model(
                slant_to_incidence_vector
            )
        if antenna_info:
            channel_model.antenna_info = translate_antenna_info_to_model(antenna_info)
        if data_statistics:
            channel_model.data_statistics = translate_data_statistics_to_model(
                data_statistics
            )
        if coreg_poly_vector:
            channel_model.coreg_poly = translate_coreg_polynomial_list_to_model(
                coreg_poly_vector
            )

        metadata_model.channel.append(channel_model)

    return metadata_model
