# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest

from arepytools.io import metadata
from arepytools.io.parsing import metadata_models as models
from arepytools.io.parsing.translate import *


class EndianityTestCase(unittest.TestCase):
    def test_translate_endianity(self):
        self.assertEqual(
            translate_endianity_from_model(models.Endianity.BIGENDIAN),
            metadata.EByteOrder.be,
        )
        self.assertEqual(
            translate_endianity_from_model(models.Endianity.LITTLEENDIAN),
            metadata.EByteOrder.le,
        )
        self.assertEqual(
            translate_endianity_to_model(metadata.EByteOrder.be),
            models.Endianity.BIGENDIAN,
        )
        self.assertEqual(
            translate_endianity_to_model(metadata.EByteOrder.le),
            models.Endianity.LITTLEENDIAN,
        )


class CellTypeTestCase(unittest.TestCase):
    def test_translate_cell_type(self):
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.FLOAT_COMPLEX),
            metadata.ECellType.fcomplex,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.FLOAT32),
            metadata.ECellType.float32,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.DOUBLE_COMPLEX),
            metadata.ECellType.dcomplex,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.FLOAT64),
            metadata.ECellType.float64,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.INT16),
            metadata.ECellType.int16,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.SHORT_COMPLEX),
            metadata.ECellType.i16complex,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.INT32),
            metadata.ECellType.int32,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.INT_COMPLEX),
            metadata.ECellType.i32complex,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.INT8),
            metadata.ECellType.int8,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.INT8_COMPLEX),
            metadata.ECellType.i8complex,
        )
        self.assertEqual(
            translate_cell_type_from_model(models.CellTypeVerboseType.CUSTOM),
            metadata.ECellType.custom,
        )

        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.fcomplex),
            models.CellTypeVerboseType.FLOAT_COMPLEX,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.float32),
            models.CellTypeVerboseType.FLOAT32,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.dcomplex),
            models.CellTypeVerboseType.DOUBLE_COMPLEX,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.float64),
            models.CellTypeVerboseType.FLOAT64,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.int16),
            models.CellTypeVerboseType.INT16,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.i16complex),
            models.CellTypeVerboseType.SHORT_COMPLEX,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.int32),
            models.CellTypeVerboseType.INT32,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.i32complex),
            models.CellTypeVerboseType.INT_COMPLEX,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.int8),
            models.CellTypeVerboseType.INT8,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.i8complex),
            models.CellTypeVerboseType.INT8_COMPLEX,
        )
        self.assertEqual(
            translate_cell_type_to_model(metadata.ECellType.custom),
            models.CellTypeVerboseType.CUSTOM,
        )


class OrbitDirectionTestCase(unittest.TestCase):
    def test_translate_orbit_direction(self):
        self.assertEqual(
            translate_orbit_direction_from_model(
                models.AscendingDescendingType.ASCENDING
            ),
            metadata.EOrbitDirection.ascending,
        )
        self.assertEqual(
            translate_orbit_direction_from_model(
                models.AscendingDescendingType.DESCENDING
            ),
            metadata.EOrbitDirection.descending,
        )
        self.assertEqual(
            translate_orbit_direction_from_model(
                models.AscendingDescendingType.NOT_AVAILABLE
            ),
            None,
        )

        self.assertEqual(
            translate_orbit_direction_to_model(metadata.EOrbitDirection.ascending),
            models.AscendingDescendingType.ASCENDING,
        )
        self.assertEqual(
            translate_orbit_direction_to_model(metadata.EOrbitDirection.descending),
            models.AscendingDescendingType.DESCENDING,
        )
        self.assertEqual(
            translate_orbit_direction_to_model(None),
            models.AscendingDescendingType.NOT_AVAILABLE,
        )


class SideLookingTestCase(unittest.TestCase):
    def test_translate_side_looking(self):
        self.assertEqual(
            translate_side_looking_from_model(models.LeftRightType.LEFT),
            metadata.ESideLooking.left_looking,
        )
        self.assertEqual(
            translate_side_looking_from_model(models.LeftRightType.RIGHT),
            metadata.ESideLooking.right_looking,
        )

        self.assertEqual(
            translate_side_looking_to_model(metadata.ESideLooking.left_looking),
            models.LeftRightType.LEFT,
        )
        self.assertEqual(
            translate_side_looking_to_model(metadata.ESideLooking.right_looking),
            models.LeftRightType.RIGHT,
        )


class PolarizationTestCase(unittest.TestCase):
    def test_translate_polarization(self):
        self.assertEqual(
            translate_polarization_from_model(models.PolarizationType.H_H),
            metadata.EPolarization.hh,
        )
        self.assertEqual(
            translate_polarization_from_model(models.PolarizationType.H_V),
            metadata.EPolarization.hv,
        )
        self.assertEqual(
            translate_polarization_from_model(models.PolarizationType.V_H),
            metadata.EPolarization.vh,
        )
        self.assertEqual(
            translate_polarization_from_model(models.PolarizationType.V_V),
            metadata.EPolarization.vv,
        )
        self.assertEqual(
            translate_polarization_from_model(models.PolarizationType.X_X),
            metadata.EPolarization.xx,
        )

        self.assertEqual(
            translate_polarization_to_model(metadata.EPolarization.hh),
            models.PolarizationType.H_H,
        )
        self.assertEqual(
            translate_polarization_to_model(metadata.EPolarization.hv),
            models.PolarizationType.H_V,
        )
        self.assertEqual(
            translate_polarization_to_model(metadata.EPolarization.vh),
            models.PolarizationType.V_H,
        )
        self.assertEqual(
            translate_polarization_to_model(metadata.EPolarization.vv),
            models.PolarizationType.V_V,
        )
        self.assertEqual(
            translate_polarization_to_model(metadata.EPolarization.xx),
            models.PolarizationType.X_X,
        )

        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.none
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.crh
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.crv
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.clh
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.clv
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.ch
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.cv
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.xh
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.xv
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.hx
        )
        self.assertRaises(
            RuntimeError, translate_polarization_to_model, metadata.EPolarization.vx
        )


class ReferenceFrameTestCase(unittest.TestCase):
    def test_translate_reference_frame(self):
        self.assertEqual(
            translate_reference_frame_from_model(models.ReferenceFrameType.GEOCENTRIC),
            metadata.EReferenceFrame.geocentric,
        )
        self.assertEqual(
            translate_reference_frame_from_model(models.ReferenceFrameType.GEODETIC),
            metadata.EReferenceFrame.geodetic,
        )
        self.assertEqual(
            translate_reference_frame_from_model(models.ReferenceFrameType.ZERODOPPLER),
            metadata.EReferenceFrame.zerodoppler,
        )

        self.assertEqual(
            translate_reference_frame_to_model(metadata.EReferenceFrame.geocentric),
            models.ReferenceFrameType.GEOCENTRIC,
        )
        self.assertEqual(
            translate_reference_frame_to_model(metadata.EReferenceFrame.geodetic),
            models.ReferenceFrameType.GEODETIC,
        )
        self.assertEqual(
            translate_reference_frame_to_model(metadata.EReferenceFrame.zerodoppler),
            models.ReferenceFrameType.ZERODOPPLER,
        )

        self.assertRaises(
            RuntimeError,
            translate_reference_frame_to_model,
            metadata.EReferenceFrame.none,
        )


class RotationOrderTestCase(unittest.TestCase):
    def test_translate_rotation_order(self):
        self.assertEqual(
            translate_rotation_order_from_model(models.RotationOrderType.YPR),
            metadata.ERotationOrder.ypr,
        )
        self.assertEqual(
            translate_rotation_order_from_model(models.RotationOrderType.YRP),
            metadata.ERotationOrder.yrp,
        )
        self.assertEqual(
            translate_rotation_order_from_model(models.RotationOrderType.RPY),
            metadata.ERotationOrder.rpy,
        )
        self.assertEqual(
            translate_rotation_order_from_model(models.RotationOrderType.RYP),
            metadata.ERotationOrder.ryp,
        )
        self.assertEqual(
            translate_rotation_order_from_model(models.RotationOrderType.PYR),
            metadata.ERotationOrder.pyr,
        )
        self.assertEqual(
            translate_rotation_order_from_model(models.RotationOrderType.PRY),
            metadata.ERotationOrder.pry,
        )

        self.assertEqual(
            translate_rotation_order_to_model(metadata.ERotationOrder.ypr),
            models.RotationOrderType.YPR,
        )

        self.assertEqual(
            translate_rotation_order_to_model(metadata.ERotationOrder.yrp),
            models.RotationOrderType.YRP,
        )

        self.assertEqual(
            translate_rotation_order_to_model(metadata.ERotationOrder.rpy),
            models.RotationOrderType.RPY,
        )

        self.assertEqual(
            translate_rotation_order_to_model(metadata.ERotationOrder.ryp),
            models.RotationOrderType.RYP,
        )

        self.assertEqual(
            translate_rotation_order_to_model(metadata.ERotationOrder.pyr),
            models.RotationOrderType.PYR,
        )

        self.assertEqual(
            translate_rotation_order_to_model(metadata.ERotationOrder.pry),
            models.RotationOrderType.PRY,
        )

        self.assertRaises(
            RuntimeError,
            translate_rotation_order_to_model,
            metadata.ERotationOrder.none,
        )


class AttitudeTypeTestCase(unittest.TestCase):
    def test_translate_attitude_type(self):
        self.assertEqual(
            translate_attitude_type_from_model(models.AttitudeType.NOMINAL),
            metadata.EAttitudeType.nominal,
        )
        self.assertEqual(
            translate_attitude_type_from_model(models.AttitudeType.REFINED),
            metadata.EAttitudeType.refined,
        )

        self.assertEqual(
            translate_attitude_type_to_model(metadata.EAttitudeType.nominal),
            models.AttitudeType.NOMINAL,
        )
        self.assertEqual(
            translate_attitude_type_to_model(metadata.EAttitudeType.refined),
            models.AttitudeType.REFINED,
        )

        self.assertRaises(
            RuntimeError,
            translate_attitude_type_to_model,
            metadata.EAttitudeType.none,
        )


class RasterFormatTypeTestCase(unittest.TestCase):
    def test_translate_raster_format_type(self):
        self.assertEqual(
            translate_raster_format_type_from_model(
                models.RasterFormatType.ARESYS_GEOTIFF
            ),
            metadata.ERasterFormatType.aresys_geotiff,
        )
        self.assertEqual(
            translate_raster_format_type_from_model(
                models.RasterFormatType.ARESYS_RASTER
            ),
            metadata.ERasterFormatType.aresys_raster,
        )
        self.assertEqual(
            translate_raster_format_type_from_model(models.RasterFormatType.RASTER),
            metadata.ERasterFormatType.aresys_raster,
        )

        self.assertEqual(
            translate_raster_format_type_to_model(
                metadata.ERasterFormatType.aresys_geotiff
            ),
            models.RasterFormatType.ARESYS_GEOTIFF,
        )

        self.assertEqual(
            translate_raster_format_type_to_model(
                metadata.ERasterFormatType.aresys_raster
            ),
            models.RasterFormatType.ARESYS_RASTER,
        )

        self.assertEqual(
            translate_raster_format_type_to_model(metadata.ERasterFormatType.raster),
            models.RasterFormatType.ARESYS_RASTER,
        )


class UnitsTestCase(unittest.TestCase):
    def test_translate_units(self):
        self.assertEqual(translate_unit_from_model(models.Units.VALUE), "")
        self.assertEqual(translate_unit_from_model(models.Units.M), "m")
        self.assertEqual(translate_unit_from_model(models.Units.S), "s")
        self.assertEqual(translate_unit_from_model(models.Units.J), "j")
        self.assertEqual(translate_unit_from_model(models.Units.D_B), "dB")
        self.assertEqual(translate_unit_from_model(models.Units.RAD), "rad")
        self.assertEqual(translate_unit_from_model(models.Units.DEG), "deg")
        self.assertEqual(translate_unit_from_model(models.Units.M_S), "m/s")
        self.assertEqual(translate_unit_from_model(models.Units.M_S2), "m/s2")
        self.assertEqual(translate_unit_from_model(models.Units.M_S3), "m/s3")
        self.assertEqual(translate_unit_from_model(models.Units.M_S4), "m/s4")
        self.assertEqual(translate_unit_from_model(models.Units.S_S), "s/s")
        self.assertEqual(translate_unit_from_model(models.Units.S_S2), "s/s2")
        self.assertEqual(translate_unit_from_model(models.Units.S_S3), "s/s3")
        self.assertEqual(translate_unit_from_model(models.Units.S_S4), "s/s4")
        self.assertEqual(translate_unit_from_model(models.Units.S_S5), "s/s5")
        self.assertEqual(translate_unit_from_model(models.Units.HZ_S), "Hz/s")
        self.assertEqual(translate_unit_from_model(models.Units.HZ_S2), "Hz/s2")
        self.assertEqual(translate_unit_from_model(models.Units.HZ_S3), "Hz/s3")
        self.assertEqual(translate_unit_from_model(models.Units.HZ_S4), "Hz/s4")
        self.assertEqual(translate_unit_from_model(models.Units.HZ_S5), "Hz/s5")
        self.assertEqual(translate_unit_from_model(models.Units.RAD_S), "rad/s")
        self.assertEqual(translate_unit_from_model(models.Units.RAD_S2), "rad/s2")
        self.assertEqual(translate_unit_from_model(models.Units.RAD_S3), "rad/s3")
        self.assertEqual(translate_unit_from_model(models.Units.RAD_S4), "rad/s4")
        self.assertEqual(translate_unit_from_model(models.Units.RAD_S5), "rad/s5")
        self.assertEqual(translate_unit_from_model(models.Units.S85), "s85")
        self.assertEqual(translate_unit_from_model(models.Units.UTC), "Utc")
        self.assertEqual(translate_unit_from_model(models.Units.B), "b")
        self.assertEqual(translate_unit_from_model(models.Units.HZ), "Hz")
        self.assertEqual(translate_unit_from_model(models.Units.K), "K")
        self.assertEqual(translate_unit_from_model(models.Units.S_M), "s/m")
        self.assertEqual(translate_unit_from_model(models.Units.S_M2), "s/m2")
        self.assertEqual(translate_unit_from_model(models.Units.S_M3), "s/m3")
        self.assertEqual(translate_unit_from_model(models.Units.S_M4), "s/m4")
        self.assertEqual(translate_unit_from_model(models.Units.DEG_S), "deg/s")
        self.assertEqual(translate_unit_from_model(models.Units.DEG_S2), "deg/s2")
        self.assertEqual(translate_unit_from_model(models.Units.DEG_S3), "deg/s3")
        self.assertEqual(translate_unit_from_model(models.Units.DEG_S4), "deg/s4")

        self.assertEqual(translate_unit_to_model(""), models.Units.VALUE)
        self.assertEqual(translate_unit_to_model("m"), models.Units.M)
        self.assertEqual(translate_unit_to_model("s"), models.Units.S)
        self.assertEqual(translate_unit_to_model("j"), models.Units.J)
        self.assertEqual(translate_unit_to_model("dB"), models.Units.D_B)
        self.assertEqual(translate_unit_to_model("rad"), models.Units.RAD)
        self.assertEqual(translate_unit_to_model("deg"), models.Units.DEG)
        self.assertEqual(translate_unit_to_model("m/s"), models.Units.M_S)
        self.assertEqual(translate_unit_to_model("m/s2"), models.Units.M_S2)
        self.assertEqual(translate_unit_to_model("m/s3"), models.Units.M_S3)
        self.assertEqual(translate_unit_to_model("m/s4"), models.Units.M_S4)
        self.assertEqual(translate_unit_to_model("s/s"), models.Units.S_S)
        self.assertEqual(translate_unit_to_model("s/s2"), models.Units.S_S2)
        self.assertEqual(translate_unit_to_model("s/s3"), models.Units.S_S3)
        self.assertEqual(translate_unit_to_model("s/s4"), models.Units.S_S4)
        self.assertEqual(translate_unit_to_model("s/s5"), models.Units.S_S5)
        self.assertEqual(translate_unit_to_model("Hz/s"), models.Units.HZ_S)
        self.assertEqual(translate_unit_to_model("Hz/s2"), models.Units.HZ_S2)
        self.assertEqual(translate_unit_to_model("Hz/s3"), models.Units.HZ_S3)
        self.assertEqual(translate_unit_to_model("Hz/s4"), models.Units.HZ_S4)
        self.assertEqual(translate_unit_to_model("Hz/s5"), models.Units.HZ_S5)
        self.assertEqual(translate_unit_to_model("rad/s"), models.Units.RAD_S)
        self.assertEqual(translate_unit_to_model("rad/s2"), models.Units.RAD_S2)
        self.assertEqual(translate_unit_to_model("rad/s3"), models.Units.RAD_S3)
        self.assertEqual(translate_unit_to_model("rad/s4"), models.Units.RAD_S4)
        self.assertEqual(translate_unit_to_model("rad/s5"), models.Units.RAD_S5)
        self.assertEqual(translate_unit_to_model("s85"), models.Units.S85)
        self.assertEqual(translate_unit_to_model("Utc"), models.Units.UTC)
        self.assertEqual(translate_unit_to_model("b"), models.Units.B)
        self.assertEqual(translate_unit_to_model("Hz"), models.Units.HZ)
        self.assertEqual(translate_unit_to_model("K"), models.Units.K)
        self.assertEqual(translate_unit_to_model("s/m"), models.Units.S_M)
        self.assertEqual(translate_unit_to_model("s/m2"), models.Units.S_M2)
        self.assertEqual(translate_unit_to_model("s/m3"), models.Units.S_M3)
        self.assertEqual(translate_unit_to_model("s/m4"), models.Units.S_M4)
        self.assertEqual(translate_unit_to_model("deg/s"), models.Units.DEG_S)
        self.assertEqual(translate_unit_to_model("deg/s2"), models.Units.DEG_S2)
        self.assertEqual(translate_unit_to_model("deg/s3"), models.Units.DEG_S3)
        self.assertEqual(translate_unit_to_model("deg/s4"), models.Units.DEG_S4)


class DoubleWithUnitTestCase(unittest.TestCase):
    def test_create_double_with_unit(self):
        self.assertEqual(
            translate_double_with_unit_to_model(5.6, "K"),
            models.DoubleWithUnit(value=5.6, unit=models.Units.K),
        )


class StringWithUnitTestCase(unittest.TestCase):
    def test_translate_str_with_unit(self):
        self.assertEqual(
            translate_str_with_unit_from_model(
                models.StringWithUnit(
                    value="01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
                )
            ),
            (PreciseDateTime.from_numeric_datetime(year=2020), "Utc"),
        )
        self.assertEqual(
            translate_str_with_unit_from_model(
                models.StringWithUnit(value="2020", unit=models.Units.M)
            ),
            (2020, "m"),
        )
        self.assertEqual(
            translate_str_with_unit_to_model(
                PreciseDateTime.from_numeric_datetime(year=2020), "Utc"
            ),
            models.StringWithUnit(
                value="01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
            ),
        )
        self.assertEqual(
            translate_str_with_unit_to_model(2020, "m"),
            models.StringWithUnit(value="2020", unit=models.Units.M),
        )


class DComplexTestCase(unittest.TestCase):
    def test_translate_dcomplex(self):
        self.assertEqual(
            translate_dcomplex_from_model(
                models.Dcomplex(real_value=2.3, imaginary_value=-4.3)
            ),
            complex(2.3, -4.3),
        )
        self.assertEqual(
            translate_dcomplex_to_model(complex(2.3, -4.3)),
            models.Dcomplex(real_value=2.3, imaginary_value=-4.3),
        )


class RasterInfoTestCase(unittest.TestCase):
    def assertEqualRasterInfo(
        self, raster_info_a: metadata.RasterInfo, raster_info_b: metadata.RasterInfo
    ):
        self.assertEqual(raster_info_a.file_name, raster_info_b.file_name)
        self.assertEqual(raster_info_a.lines, raster_info_b.lines)
        self.assertEqual(raster_info_a.samples, raster_info_b.samples)
        self.assertEqual(
            raster_info_a.header_offset_bytes, raster_info_b.header_offset_bytes
        )
        self.assertEqual(raster_info_a.row_prefix_bytes, raster_info_b.row_prefix_bytes)
        self.assertEqual(raster_info_a.lines_start, raster_info_b.lines_start)
        self.assertEqual(raster_info_a.lines_start_unit, raster_info_b.lines_start_unit)
        self.assertEqual(raster_info_a.lines_step, raster_info_b.lines_step)
        self.assertEqual(raster_info_a.lines_step_unit, raster_info_b.lines_step_unit)
        self.assertEqual(raster_info_a.samples_start, raster_info_b.samples_start)
        self.assertEqual(
            raster_info_a.samples_start_unit, raster_info_b.samples_start_unit
        )
        self.assertEqual(raster_info_a.samples_step, raster_info_b.samples_step)
        self.assertEqual(
            raster_info_a.samples_step_unit, raster_info_b.samples_step_unit
        )
        self.assertEqual(raster_info_a.byte_order, raster_info_b.byte_order)
        self.assertEqual(raster_info_a.cell_type, raster_info_b.cell_type)
        self.assertEqual(raster_info_a.invalid_value, raster_info_b.invalid_value)
        self.assertEqual(raster_info_a.format_type, raster_info_b.format_type)

    def setUp(self) -> None:
        self.raster_info_model = models.RasterInfoType(
            file_name="filename.tiff",
            lines=11,
            samples=3,
            header_offset_bytes=50,
            row_prefix_bytes=4,
            byte_order=models.Endianity.BIGENDIAN,
            cell_type=models.CellTypeVerboseType.INT16,
            lines_step=translate_double_with_unit_to_model(0.5, "Hz"),
            samples_step=translate_double_with_unit_to_model(8.5, "m"),
            invalid_value=None,
            lines_start=translate_str_with_unit_to_model(
                PreciseDateTime.from_numeric_datetime(year=2020), "Utc"
            ),
            samples_start=translate_str_with_unit_to_model(3.4, "m"),
            raster_format=None,
        )
        self.raster_info_metadata = metadata.RasterInfo(
            lines=11,
            samples=3,
            celltype=metadata.ECellType.int16,
            filename="filename.tiff",
            header_offset_bytes=50,
            row_prefix_bytes=4,
            byteorder=metadata.EByteOrder.be,  # type:ignore
            invalid_value=None,
            format_type=None,
        )
        self.raster_info_metadata.set_lines_axis(
            PreciseDateTime.from_numeric_datetime(year=2020), "Utc", 0.5, "Hz"
        )
        self.raster_info_metadata.set_samples_axis(3.4, "m", 8.5, "m")

    def test_translate_raster_info(self):
        self.assertEqualRasterInfo(
            translate_raster_info_from_model(self.raster_info_model),
            self.raster_info_metadata,
        )

        self.assertEqual(
            translate_raster_info_to_model(self.raster_info_metadata),
            self.raster_info_model,
        )

    def test_translate_raster_info_range_utc_lines_float(self):
        self.raster_info_model.lines_start = translate_str_with_unit_to_model(
            1000.4, "Hz"
        )
        self.raster_info_model.samples_start = translate_str_with_unit_to_model(
            PreciseDateTime.from_numeric_datetime(year=2020), "Utc"
        )
        self.raster_info_metadata.set_lines_axis(
            lines_start=1000.4,
            lines_start_unit="Hz",
            lines_step=self.raster_info_metadata.lines_step,
            lines_step_unit=self.raster_info_metadata.lines_step_unit,
        )
        self.raster_info_metadata.set_samples_axis(
            samples_start=PreciseDateTime.from_numeric_datetime(year=2020),
            samples_start_unit="Utc",
            samples_step=self.raster_info_metadata.samples_step,
            samples_step_unit=self.raster_info_metadata.samples_step_unit,
        )
        self.assertEqualRasterInfo(
            translate_raster_info_from_model(self.raster_info_model),
            self.raster_info_metadata,
        )

        self.assertEqual(
            translate_raster_info_to_model(self.raster_info_metadata),
            self.raster_info_model,
        )

    def test_translate_raster_info_invalid_value_and_format(self):
        self.raster_info_model.invalid_value = models.Dcomplex(2.3, -7.8)
        self.raster_info_metadata._invalid_value = complex(2.3, -7.8)

        self.raster_info_model.raster_format = models.RasterFormatType.ARESYS_GEOTIFF
        self.raster_info_metadata._format_type = (
            metadata.ERasterFormatType.aresys_geotiff
        )

        self.assertEqualRasterInfo(
            translate_raster_info_from_model(self.raster_info_model),
            self.raster_info_metadata,
        )

        self.assertEqual(
            translate_raster_info_to_model(self.raster_info_metadata),
            self.raster_info_model,
        )


class DataSetInfoTestCase(unittest.TestCase):
    def assertEqualDataSetInfo(
        self, info_a: metadata.DataSetInfo, info_b: metadata.DataSetInfo
    ):
        self.assertEqual(info_a.sensor_name, info_b.sensor_name)
        self.assertEqual(info_a.description, info_b.description)
        self.assertEqual(info_a.acquisition_mode, info_b.acquisition_mode)
        self.assertEqual(info_a.image_type, info_b.image_type)
        self.assertEqual(info_a.projection, info_b.projection)
        self.assertEqual(info_a.acquisition_station, info_b.acquisition_station)
        self.assertEqual(info_a.processing_center, info_b.processing_center)
        self.assertEqual(info_a.processing_software, info_b.processing_software)
        self.assertEqual(info_a.fc_hz, info_b.fc_hz)
        self.assertEqual(
            info_a.external_calibration_factor, info_b.external_calibration_factor
        )
        self.assertEqual(info_a.data_take_id, info_b.data_take_id)
        self.assertEqual(info_a.sense_date, info_b.sense_date)
        self.assertEqual(info_a.processing_date, info_b.processing_date)
        self.assertEqual(info_a.side_looking, info_b.side_looking)

    def setUp(self) -> None:
        self.data_set_info_model = models.DataSetInfoType(
            sensor_name="sensor",
            description=models.DataSetInfoType.Description("description"),
            sense_date=models.DataSetInfoType.SenseDate("NOT_AVAILABLE"),
            acquisition_mode=models.DataSetInfoType.AcquisitionMode("acquisition_mode"),
            image_type=models.DataSetInfoType.ImageType("image_type"),
            projection=models.DataSetInfoType.Projection("projection"),
            acquisition_station=models.DataSetInfoType.AcquisitionStation(
                "acquisition_station"
            ),
            processing_center=models.DataSetInfoType.ProcessingCenter(
                "processing_center"
            ),
            processing_date=models.DataSetInfoType.ProcessingDate("NOT_AVAILABLE"),
            processing_software=models.DataSetInfoType.ProcessingSoftware(
                "processing_software"
            ),
            fc_hz=models.DataSetInfoType.FcHz(1000),
            side_looking=models.LeftRightType.LEFT,
            external_calibration_factor=None,
            data_take_id=None,
        )
        self.data_set_info_metadata = metadata.DataSetInfo(
            acquisition_mode_i="acquisition_mode", fc_hz_i=1000
        )
        self.data_set_info_metadata.sensor_name = "sensor"
        self.data_set_info_metadata.description = "description"
        self.data_set_info_metadata.sense_date = None
        self.data_set_info_metadata.acquisition_mode = "acquisition_mode"
        self.data_set_info_metadata.image_type = "image_type"
        self.data_set_info_metadata.projection = "projection"
        self.data_set_info_metadata.acquisition_station = "acquisition_station"
        self.data_set_info_metadata.processing_center = "processing_center"
        self.data_set_info_metadata.processing_date = None
        self.data_set_info_metadata.processing_software = "processing_software"
        self.data_set_info_metadata.side_looking = metadata.ESideLooking.left_looking

    def test_translate_data_set_info(self):
        self.assertEqualDataSetInfo(
            translate_dataset_info_from_model(self.data_set_info_model),
            self.data_set_info_metadata,
        )

        self.assertEqual(
            translate_dataset_info_to_model(self.data_set_info_metadata),
            self.data_set_info_model,
        )

    def test_translate_data_set_info_with_dates(self):
        assert self.data_set_info_model.sense_date is not None
        self.data_set_info_model.sense_date.value = "01-JAN-2020 00:00:00.000000000000"
        self.data_set_info_metadata.sense_date = PreciseDateTime.from_numeric_datetime(
            year=2020
        )
        assert self.data_set_info_model.processing_date is not None
        self.data_set_info_model.processing_date.value = (
            "01-JAN-2021 00:00:00.000000000000"
        )
        self.data_set_info_metadata.processing_date = (
            PreciseDateTime.from_numeric_datetime(year=2021)
        )

        self.assertEqualDataSetInfo(
            translate_dataset_info_from_model(self.data_set_info_model),
            self.data_set_info_metadata,
        )

        self.assertEqual(
            translate_dataset_info_to_model(self.data_set_info_metadata),
            self.data_set_info_model,
        )

    def test_translate_data_set_info_with_additional_info(self):
        self.data_set_info_model.external_calibration_factor = 15.2
        self.data_set_info_model.data_take_id = 20
        self.data_set_info_metadata.external_calibration_factor = 15.2
        self.data_set_info_metadata.data_take_id = 20

        self.assertEqualDataSetInfo(
            translate_dataset_info_from_model(self.data_set_info_model),
            self.data_set_info_metadata,
        )

        self.assertEqual(
            translate_dataset_info_to_model(self.data_set_info_metadata),
            self.data_set_info_model,
        )


def assertEqualGeoPoint(
    testCase: unittest.TestCase, point_a: metadata.GeoPoint, point_b: metadata.GeoPoint
):
    """Assert that geo point are equal"""
    testCase.assertEqual(point_a.lat, point_b.lat)
    testCase.assertEqual(point_a.lon, point_b.lon)
    testCase.assertEqual(point_a.height, point_b.height)
    testCase.assertEqual(point_a.theta_inc, point_b.theta_inc)
    testCase.assertEqual(point_a.theta_look, point_b.theta_look)


class GeoPointTestCase(unittest.TestCase):
    def test_translate_geo_point(self):
        point_model = models.PointType(
            [
                models.PointType.Val(0.3),
                models.PointType.Val(1.2),
                models.PointType.Val(2.1),
                models.PointType.Val(3.9),
                models.PointType.Val(4.8),
            ]
        )
        point_metadata = metadata.GeoPoint(
            lat=0.3, lon=1.2, height=2.1, theta_inc=3.9, theta_look=4.8
        )

        assertEqualGeoPoint(
            self, translate_geo_point_from_model(point_model), point_metadata
        )
        self.assertEqual(translate_geo_point_to_model(point_metadata), point_model)


class GroundCornerPointsTestCase(unittest.TestCase):
    def assertEqualGroundCornerPoints(
        self,
        points_a: metadata.GroundCornerPoints,
        points_b: metadata.GroundCornerPoints,
    ):
        self.assertEqual(points_a.easting_grid_size, points_b.easting_grid_size)
        self.assertEqual(points_a.northing_grid_size, points_b.northing_grid_size)
        assertEqualGeoPoint(self, points_a.center_point, points_b.center_point)
        assertEqualGeoPoint(self, points_a.ne_point, points_b.ne_point)
        assertEqualGeoPoint(self, points_a.nw_point, points_b.nw_point)
        assertEqualGeoPoint(self, points_a.se_point, points_b.se_point)
        assertEqualGeoPoint(self, points_a.sw_point, points_b.sw_point)

    def test_translate_ground_corner_points(self):
        point_0_model = models.PointType(
            [
                models.PointType.Val(0.0),
                models.PointType.Val(0.0),
                models.PointType.Val(0.0),
                models.PointType.Val(0.0),
                models.PointType.Val(0.0),
            ]
        )
        point_1_model = models.PointType(
            [
                models.PointType.Val(1.0),
                models.PointType.Val(1.0),
                models.PointType.Val(1.0),
                models.PointType.Val(1.0),
                models.PointType.Val(1.0),
            ]
        )
        point_2_model = models.PointType(
            [
                models.PointType.Val(2.0),
                models.PointType.Val(2.0),
                models.PointType.Val(2.0),
                models.PointType.Val(2.0),
                models.PointType.Val(2.0),
            ]
        )
        point_3_model = models.PointType(
            [
                models.PointType.Val(3.0),
                models.PointType.Val(3.0),
                models.PointType.Val(3.0),
                models.PointType.Val(3.0),
                models.PointType.Val(3.0),
            ]
        )
        point_4_model = models.PointType(
            [
                models.PointType.Val(4.0),
                models.PointType.Val(4.0),
                models.PointType.Val(4.0),
                models.PointType.Val(4.0),
                models.PointType.Val(4.0),
            ]
        )

        corners_model = models.GroundCornersPointsType(
            easting_grid_size=models.GroundCornersPointsType.EastingGridSize(5.3),
            northing_grid_size=models.GroundCornersPointsType.NorthingGridSize(8.1),
            north_west=models.GroundCornersPointsType.NorthWest(point_0_model),
            north_east=models.GroundCornersPointsType.NorthEast(point_1_model),
            south_west=models.GroundCornersPointsType.SouthWest(point_2_model),
            south_east=models.GroundCornersPointsType.SouthEast(point_3_model),
            center=models.GroundCornersPointsType.Center(point_4_model),
        )

        corners_metadata = metadata.GroundCornerPoints()
        corners_metadata.easting_grid_size = 5.3
        corners_metadata.northing_grid_size = 8.1
        corners_metadata.nw_point = translate_geo_point_from_model(point_0_model)
        corners_metadata.ne_point = translate_geo_point_from_model(point_1_model)
        corners_metadata.sw_point = translate_geo_point_from_model(point_2_model)
        corners_metadata.se_point = translate_geo_point_from_model(point_3_model)
        corners_metadata.center_point = translate_geo_point_from_model(point_4_model)

        self.assertEqualGroundCornerPoints(
            translate_ground_corner_points_from_model(corners_model), corners_metadata
        )
        self.assertEqual(
            translate_ground_corner_points_to_model(corners_metadata), corners_model
        )


class SwathInfoTestCase(unittest.TestCase):
    def assertEqualSwathInfo(
        self, info_a: metadata.SwathInfo, info_b: metadata.SwathInfo
    ):
        self.assertEqual(info_a.swath, info_b.swath)
        self.assertEqual(info_a.polarization, info_b.polarization)
        self.assertEqual(info_a.acquisition_prf, info_b.acquisition_prf)
        self.assertEqual(info_a.acquisition_prf_unit, info_b.acquisition_prf_unit)
        self.assertEqual(info_a.swath_acquisition_order, info_b.swath_acquisition_order)
        self.assertEqual(info_a.rank, info_b.rank)
        self.assertEqual(info_a.range_delay_bias, info_b.range_delay_bias)
        self.assertEqual(info_a.range_delay_bias_unit, info_b.range_delay_bias_unit)
        self.assertEqual(info_a.acquisition_start_time, info_b.acquisition_start_time)
        self.assertEqual(
            info_a.acquisition_start_time_unit, info_b.acquisition_start_time_unit
        )
        self.assertEqual(
            info_a.azimuth_steering_rate_reference_time,
            info_b.azimuth_steering_rate_reference_time,
        )
        self.assertEqual(
            info_a.az_steering_rate_ref_time_unit, info_b.az_steering_rate_ref_time_unit
        )
        self.assertEqual(info_a.echoes_per_burst, info_b.echoes_per_burst)
        self.assertEqual(
            info_a.azimuth_steering_rate_pol, info_b.azimuth_steering_rate_pol
        )
        self.assertEqual(info_a.rx_gain, info_b.rx_gain)
        self.assertEqual(info_a.channel_delay, info_b.channel_delay)

    def test_translate_swath_info(self):
        swath_info_model = models.SwathInfoType(
            swath=models.SwathInfoType.Swath("swath_name"),
            swath_acquisition_order=models.SwathInfoType.SwathAcquisitionOrder(4),
            polarization=models.PolarizationType.H_H,
            rank=models.SwathInfoType.Rank(15),
            range_delay_bias=models.SwathInfoType.RangeDelayBias(
                0.5, unit=models.Units.S
            ),
            acquisition_start_time=models.SwathInfoType.AcquisitionStartTime(
                "01-JAN-2020 00:00:00.000000000000", models.Units.UTC
            ),
            azimuth_steering_rate_reference_time=models.DoubleWithUnit(
                1.0, models.Units.S
            ),
            azimuth_steering_rate_pol=models.SwathInfoType.AzimuthSteeringRatePol(
                [
                    models.SwathInfoType.AzimuthSteeringRatePol.Val(0.1, n=1),
                    models.SwathInfoType.AzimuthSteeringRatePol.Val(0.2, n=2),
                    models.SwathInfoType.AzimuthSteeringRatePol.Val(0.3, n=3),
                ]
            ),
            acquisition_prf=1000.0,
            echoes_per_burst=153,
            channel_delay=0.5,
            rx_gain=85.2,
        )

        swath_info_metadata = metadata.SwathInfo(
            swath_i="swath_name",
            polarization_i=metadata.EPolarization.hh,
            acquisition_prf_i=1000.0,
        )
        swath_info_metadata.swath_acquisition_order = 4
        swath_info_metadata.rank = 15
        swath_info_metadata.range_delay_bias = 0.5
        swath_info_metadata.acquisition_start_time = PreciseDateTime.from_utc_string(
            "01-JAN-2020 00:00:00.000000000000"
        )
        swath_info_metadata.azimuth_steering_rate_reference_time = 1.0
        swath_info_metadata.azimuth_steering_rate_pol = (0.1, 0.2, 0.3)
        swath_info_metadata.echoes_per_burst = 153
        swath_info_metadata.channel_delay = 0.5
        swath_info_metadata.rx_gain = 85.2

        self.assertEqualSwathInfo(
            translate_swath_info_from_model(swath_info_model), swath_info_metadata
        )

        self.assertEqual(
            translate_swath_info_to_model(swath_info_metadata), swath_info_model
        )


class SamplingConstantsTestCase(unittest.TestCase):
    def assertEqualSamplingConstants(
        self,
        constants_a: metadata.SamplingConstants,
        constants_b: metadata.SamplingConstants,
    ):
        self.assertEqual(constants_a.frg_hz, constants_b.frg_hz)
        self.assertEqual(constants_a.brg_hz, constants_b.brg_hz)
        self.assertEqual(constants_a.faz_hz, constants_b.faz_hz)
        self.assertEqual(constants_a.baz_hz, constants_b.baz_hz)

    def test_translate_sampling_constants(self):
        constants_model = models.SamplingConstantsType(
            frg_hz=models.SamplingConstantsType.FrgHz(150.0, unit=models.Units.HZ),
            brg_hz=models.SamplingConstantsType.BrgHz(120.0, unit=models.Units.HZ),
            faz_hz=models.SamplingConstantsType.FazHz(2000.0, unit=models.Units.HZ),
            baz_hz=models.SamplingConstantsType.BazHz(1500.0, unit=models.Units.HZ),
        )

        constants_metadata = metadata.SamplingConstants(
            frg_hz_i=150.0, brg_hz_i=120.0, faz_hz_i=2000.0, baz_hz_i=1500.0
        )

        self.assertEqualSamplingConstants(
            translate_sampling_constants_from_model(constants_model), constants_metadata
        )

        self.assertEqual(
            translate_sampling_constants_to_model(constants_metadata), constants_model
        )


class AcquisitionTimeLineTestCase(unittest.TestCase):
    def assertEqualAcquisitionTimeLine(
        self,
        time_line_a: metadata.AcquisitionTimeLine,
        time_line_b: metadata.AcquisitionTimeLine,
    ):
        self.assertEqual(time_line_a.missing_lines, time_line_b.missing_lines)
        self.assertEqual(time_line_a.swst_changes, time_line_b.swst_changes)
        self.assertEqual(time_line_a.noise_packet, time_line_b.noise_packet)
        self.assertEqual(
            time_line_a.internal_calibration, time_line_b.internal_calibration
        )
        self.assertEqual(time_line_a.swl_changes, time_line_b.swl_changes)
        self.assertEqual(time_line_a.prf_changes, time_line_b.prf_changes)
        self.assertEqual(time_line_a.duplicated_lines, time_line_b.duplicated_lines)
        self.assertEqual(time_line_a.chirp_period, time_line_b.chirp_period)

    def setUp(self) -> None:
        self.time_line_metadata = metadata.AcquisitionTimeLine()
        self.time_line_model = models.AcquisitionTimelineType(
            missing_lines_number=0,
            missing_lines_azimuthtimes=models.AcquisitionTimelineType.MissingLinesAzimuthtimes(
                []
            ),
            swst_changes_number=0,
            swst_changes_azimuthtimes=models.AcquisitionTimelineType.SwstChangesAzimuthtimes(
                []
            ),
            swst_changes_values=models.AcquisitionTimelineType.SwstChangesValues([]),
            noise_packets_number=0,
            noise_packets_azimuthtimes=models.AcquisitionTimelineType.NoisePacketsAzimuthtimes(
                []
            ),
            internal_calibration_number=0,
            internal_calibration_azimuthtimes=models.AcquisitionTimelineType.InternalCalibrationAzimuthtimes(
                []
            ),
        )

    def test_translate_acquisition_time_line(self):
        self.assertEqual(
            translate_acquisition_time_line_to_model(self.time_line_metadata),
            self.time_line_model,
        )

        self.assertEqualAcquisitionTimeLine(
            translate_acquisition_time_line_from_model(self.time_line_model),
            self.time_line_metadata,
        )

    def test_translate_acquisition_time_line_with_changes(self):
        self.time_line_metadata.missing_lines = [0.0, 1.1, 2.1]
        self.time_line_metadata.swst_changes = (2, [0.0, 1.2], [0.005, 0.0051])
        self.time_line_metadata.noise_packet = [0.0, 5.5, 15.2, 16.2]
        self.time_line_metadata.internal_calibration = [0.0, 2.3, 4.8, 6.9, 15.2]

        self.time_line_model.missing_lines_number = 3
        self.time_line_model.missing_lines_azimuthtimes = (
            models.AcquisitionTimelineType.MissingLinesAzimuthtimes(
                [
                    models.AcquisitionTimelineType.MissingLinesAzimuthtimes.Val(
                        0.0, models.Units.S
                    ),
                    models.AcquisitionTimelineType.MissingLinesAzimuthtimes.Val(
                        1.1, models.Units.S
                    ),
                    models.AcquisitionTimelineType.MissingLinesAzimuthtimes.Val(
                        2.1, models.Units.S
                    ),
                ]
            )
        )
        self.time_line_model.swst_changes_number = 2
        self.time_line_model.swst_changes_azimuthtimes = (
            models.AcquisitionTimelineType.SwstChangesAzimuthtimes(
                [
                    models.AcquisitionTimelineType.SwstChangesAzimuthtimes.Val(
                        0.0, models.Units.S
                    ),
                    models.AcquisitionTimelineType.SwstChangesAzimuthtimes.Val(
                        1.2, models.Units.S
                    ),
                ]
            )
        )
        self.time_line_model.swst_changes_values = (
            models.AcquisitionTimelineType.SwstChangesValues(
                [
                    models.AcquisitionTimelineType.SwstChangesValues.Val(
                        0.005, models.Units.S
                    ),
                    models.AcquisitionTimelineType.SwstChangesValues.Val(
                        0.0051, models.Units.S
                    ),
                ]
            )
        )
        self.time_line_model.noise_packets_number = 4
        self.time_line_model.noise_packets_azimuthtimes = (
            models.AcquisitionTimelineType.NoisePacketsAzimuthtimes(
                [
                    models.AcquisitionTimelineType.NoisePacketsAzimuthtimes.Val(
                        0.0, unit=models.Units.S
                    ),
                    models.AcquisitionTimelineType.NoisePacketsAzimuthtimes.Val(
                        5.5, unit=models.Units.S
                    ),
                    models.AcquisitionTimelineType.NoisePacketsAzimuthtimes.Val(
                        15.2, unit=models.Units.S
                    ),
                    models.AcquisitionTimelineType.NoisePacketsAzimuthtimes.Val(
                        16.2, unit=models.Units.S
                    ),
                ]
            )
        )
        self.time_line_model.internal_calibration_number = 5

        self.time_line_model.internal_calibration_azimuthtimes = (
            models.AcquisitionTimelineType.InternalCalibrationAzimuthtimes(
                [
                    models.AcquisitionTimelineType.InternalCalibrationAzimuthtimes.Val(
                        0.0, unit=models.Units.S
                    ),
                    models.AcquisitionTimelineType.InternalCalibrationAzimuthtimes.Val(
                        2.3, unit=models.Units.S
                    ),
                    models.AcquisitionTimelineType.InternalCalibrationAzimuthtimes.Val(
                        4.8, unit=models.Units.S
                    ),
                    models.AcquisitionTimelineType.InternalCalibrationAzimuthtimes.Val(
                        6.9, unit=models.Units.S
                    ),
                    models.AcquisitionTimelineType.InternalCalibrationAzimuthtimes.Val(
                        15.2, unit=models.Units.S
                    ),
                ]
            )
        )

        self.assertEqual(
            translate_acquisition_time_line_to_model(self.time_line_metadata),
            self.time_line_model,
        )

        self.assertEqualAcquisitionTimeLine(
            translate_acquisition_time_line_from_model(self.time_line_model),
            self.time_line_metadata,
        )

    def test_translate_acquisition_time_line_with_optional_changes(self):
        self.time_line_metadata.duplicated_lines = (3, [0.0, 1.1, 2.1])
        self.time_line_metadata.swl_changes = (2, [0.0, 1.2], [0.005, 0.0051])
        self.time_line_metadata.prf_changes = (
            4,
            [0.0, 2.0, 3.0, 4.0],
            [1000.0, 1200.0, 1400.0, 1500.0],
        )
        self.time_line_metadata.chirp_period = "Chirp period"

        self.time_line_model.duplicated_lines_number = 3
        self.time_line_model.duplicated_lines_azimuthtimes = (
            models.AcquisitionTimelineType.DuplicatedLinesAzimuthtimes(
                [
                    models.AcquisitionTimelineType.DuplicatedLinesAzimuthtimes.Val(
                        0.0, models.Units.S
                    ),
                    models.AcquisitionTimelineType.DuplicatedLinesAzimuthtimes.Val(
                        1.1, models.Units.S
                    ),
                    models.AcquisitionTimelineType.DuplicatedLinesAzimuthtimes.Val(
                        2.1, models.Units.S
                    ),
                ]
            )
        )

        self.time_line_model.swl_changes_number = 2
        self.time_line_model.swl_changes_azimuthtimes = (
            models.AcquisitionTimelineType.SwlChangesAzimuthtimes(
                [
                    models.AcquisitionTimelineType.SwlChangesAzimuthtimes.Val(
                        0.0, models.Units.S
                    ),
                    models.AcquisitionTimelineType.SwlChangesAzimuthtimes.Val(
                        1.2, models.Units.S
                    ),
                ]
            )
        )
        self.time_line_model.swl_changes_values = (
            models.AcquisitionTimelineType.SwlChangesValues(
                [
                    models.AcquisitionTimelineType.SwlChangesValues.Val(
                        0.005, models.Units.S
                    ),
                    models.AcquisitionTimelineType.SwlChangesValues.Val(
                        0.0051, models.Units.S
                    ),
                ]
            )
        )

        self.time_line_model.prf_changes_number = 4
        self.time_line_model.prf_changes_azimuthtimes = (
            models.AcquisitionTimelineType.PrfChangesAzimuthtimes(
                [
                    models.AcquisitionTimelineType.PrfChangesAzimuthtimes.Val(
                        0.0, models.Units.S
                    ),
                    models.AcquisitionTimelineType.PrfChangesAzimuthtimes.Val(
                        2.0, models.Units.S
                    ),
                    models.AcquisitionTimelineType.PrfChangesAzimuthtimes.Val(
                        3.0, models.Units.S
                    ),
                    models.AcquisitionTimelineType.PrfChangesAzimuthtimes.Val(
                        4.0, models.Units.S
                    ),
                ]
            )
        )
        self.time_line_model.prf_changes_values = (
            models.AcquisitionTimelineType.PrfChangesValues(
                [
                    models.AcquisitionTimelineType.PrfChangesValues.Val(
                        1000.0, models.Units.HZ
                    ),
                    models.AcquisitionTimelineType.PrfChangesValues.Val(
                        1200.0, models.Units.HZ
                    ),
                    models.AcquisitionTimelineType.PrfChangesValues.Val(
                        1400.0, models.Units.HZ
                    ),
                    models.AcquisitionTimelineType.PrfChangesValues.Val(
                        1500.0, models.Units.HZ
                    ),
                ]
            )
        )

        self.time_line_model.chirp_period = "Chirp period"

        self.assertEqual(
            translate_acquisition_time_line_to_model(self.time_line_metadata),
            self.time_line_model,
        )

        self.assertEqualAcquisitionTimeLine(
            translate_acquisition_time_line_from_model(self.time_line_model),
            self.time_line_metadata,
        )


class AttitudeInfoTestCase(unittest.TestCase):
    def assertEqualAttitudeInfo(
        self, attitude_a: metadata.AttitudeInfo, attitude_b: metadata.AttitudeInfo
    ):
        self.assertEqual(attitude_a.reference_frame, attitude_b.reference_frame)
        self.assertEqual(attitude_a.rotation_order, attitude_b.rotation_order)
        self.assertEqual(
            attitude_a.attitude_records_number, attitude_b.attitude_records_number
        )
        self.assertEqual(attitude_a.attitude_type, attitude_b.attitude_type)

        self.assertEqual(len(attitude_a.yaw_vector), len(attitude_a.yaw_vector))
        for yaw_a, yaw_b in zip(attitude_a.yaw_vector, attitude_b.yaw_vector):
            self.assertEqual(yaw_a, yaw_b)

        self.assertEqual(len(attitude_a.pitch_vector), len(attitude_a.pitch_vector))
        for pitch_a, pitch_b in zip(attitude_a.pitch_vector, attitude_b.pitch_vector):
            self.assertEqual(pitch_a, pitch_b)

            self.assertEqual(len(attitude_a.roll_vector), len(attitude_a.roll_vector))
        for roll_a, roll_b in zip(attitude_a.roll_vector, attitude_b.roll_vector):
            self.assertEqual(roll_a, roll_b)

        self.assertEqual(attitude_a.reference_time, attitude_b.reference_time)
        self.assertEqual(attitude_a.time_step, attitude_b.time_step)

    def setUp(self) -> None:
        self.attitude_model = models.AttitudeInfoType(
            t_ref_utc="01-JAN-2023 00:00:00.000000000000",
            dt_ypr_s=models.AttitudeInfoType.DtYprS(1.5),
            n_ypr_n=models.AttitudeInfoType.NYprN(3),
            yaw_deg=models.AttitudeInfoType.YawDeg(
                [
                    models.AttitudeInfoType.YawDeg.Val(0.5, n=1),
                    models.AttitudeInfoType.YawDeg.Val(1.0, n=2),
                    models.AttitudeInfoType.YawDeg.Val(1.5, n=3),
                ]
            ),
            pitch_deg=models.AttitudeInfoType.PitchDeg(
                [
                    models.AttitudeInfoType.PitchDeg.Val(-0.5, n=1),
                    models.AttitudeInfoType.PitchDeg.Val(-1.0, n=2),
                    models.AttitudeInfoType.PitchDeg.Val(-1.5, n=3),
                ]
            ),
            roll_deg=models.AttitudeInfoType.RollDeg(
                [
                    models.AttitudeInfoType.RollDeg.Val(30.2, n=1),
                    models.AttitudeInfoType.RollDeg.Val(31.2, n=2),
                    models.AttitudeInfoType.RollDeg.Val(32.2, n=3),
                ]
            ),
            reference_frame=models.ReferenceFrameType.ZERODOPPLER,
            rotation_order=models.RotationOrderType.PRY,
            attitude_type=models.AttitudeType.REFINED,
        )

        self.attitude_metadata = metadata.AttitudeInfo(
            yaw=[0.5, 1.0, 1.5],
            pitch=[-0.5, -1.0, -1.5],
            roll=[30.2, 31.2, 32.2],
            t0=PreciseDateTime.from_numeric_datetime(year=2023),
            delta_t=1.5,
            ref_frame=metadata.EReferenceFrame.zerodoppler.value,
            rot_order=metadata.ERotationOrder.pry.value,
        )
        self.attitude_metadata.attitude_type = metadata.EAttitudeType.refined.value

    def test_translate_attitude_info(self):
        self.assertEqualAttitudeInfo(
            translate_attitude_from_model(self.attitude_model), self.attitude_metadata
        )

        self.assertEqual(
            translate_attitude_to_model(self.attitude_metadata), self.attitude_model
        )


class BurstInfoTestCase(unittest.TestCase):
    def assertEqualBurst(self, burst_a: metadata.Burst, burst_b: metadata.Burst):
        self.assertEqual(burst_a.range_start_time, burst_b.range_start_time)
        self.assertEqual(burst_a.azimuth_start_time, burst_b.azimuth_start_time)
        self.assertEqual(
            burst_a.burst_center_azimuth_shift, burst_b.burst_center_azimuth_shift
        )
        self.assertEqual(burst_a.lines, burst_b.lines)

    def assertEqualBurstInfo(
        self, info_a: metadata.BurstInfo, info_b: metadata.BurstInfo
    ):
        self.assertEqual(
            info_a.burst_repetition_frequency, info_b.burst_repetition_frequency
        )
        self.assertEqual(
            info_a.is_lines_per_burst_present(), info_b.is_lines_per_burst_present()
        )
        self.assertEqual(info_a.get_number_of_bursts(), info_b.get_number_of_bursts())
        for index in range(info_a.get_number_of_bursts()):
            self.assertEqualBurst(info_a.get_burst(index), info_b.get_burst(index))

    def setUp(self):
        self.burst_info_metadata = metadata.BurstInfo(burst_repetition_frequency=20.5)
        self.burst_info_metadata.add_burst(
            range_start_time_i=0.005,
            azimuth_start_time_i=PreciseDateTime.from_numeric_datetime(year=2021),
            lines_i=1520,
        )
        self.burst_info_metadata.add_burst(
            range_start_time_i=0.0055,
            azimuth_start_time_i=PreciseDateTime.from_numeric_datetime(year=2023),
            lines_i=1520,
        )

        burst_model_one = models.BurstType(
            range_start_time=models.DoubleWithUnit(0.005, unit=models.Units.S),
            azimuth_start_time=models.StringWithUnit(
                "01-JAN-2021 00:00:00.000000000000", unit=models.Units.UTC
            ),
            n=1,
        )
        burst_model_two = models.BurstType(
            range_start_time=models.DoubleWithUnit(0.0055, unit=models.Units.S),
            azimuth_start_time=models.StringWithUnit(
                "01-JAN-2023 00:00:00.000000000000", unit=models.Units.UTC
            ),
            n=2,
        )

        self.burst_info_model = models.BurstInfoType(
            number_of_bursts=2,
            lines_per_burst=None,
            lines_per_burst_change_list=models.BurstInfoType.LinesPerBurstChangeList(
                [models.BurstInfoType.LinesPerBurstChangeList.Lines(1520, 1)]
            ),
            burst_repetition_frequency=models.DoubleWithUnit(20.5, models.Units.HZ),
            burst=[burst_model_one, burst_model_two],
        )

    def test_translate_burst_info(self):
        self.assertEqualBurstInfo(
            translate_burst_info_from_model(self.burst_info_model),
            self.burst_info_metadata,
        )

        self.assertEqual(
            translate_burst_info_to_model(self.burst_info_metadata),
            self.burst_info_model,
        )

    def test_translate_burst_info_different_lines_and_shift(self):
        self.burst_info_model.burst.append(
            models.BurstType(
                range_start_time=models.DoubleWithUnit(0.008, unit=models.Units.S),
                azimuth_start_time=models.StringWithUnit(
                    "01-JAN-2022 00:00:00.000000000000", unit=models.Units.UTC
                ),
                n=3,
                burst_center_azimuth_shift=models.DoubleWithUnit(0.56, models.Units.S),
            )
        )
        self.burst_info_model.number_of_bursts = 3
        assert self.burst_info_model.lines_per_burst_change_list is not None
        self.burst_info_model.lines_per_burst_change_list.lines.append(
            models.BurstInfoType.LinesPerBurstChangeList.Lines(2000, 3)
        )

        self.burst_info_metadata.add_burst(
            range_start_time_i=0.008,
            azimuth_start_time_i=PreciseDateTime.from_numeric_datetime(year=2022),
            lines_i=2000,
            burst_center_azimuth_shift_i=0.56,
        )

        self.assertEqualBurstInfo(
            translate_burst_info_from_model(self.burst_info_model),
            self.burst_info_metadata,
        )

        self.assertEqual(
            translate_burst_info_to_model(self.burst_info_metadata),
            self.burst_info_model,
        )


class StateVectorsTestCase(unittest.TestCase):
    def assertEqualStateVectors(
        self, sv_a: metadata.StateVectors, sv_b: metadata.StateVectors
    ):
        self.assertEqual(sv_a.anx_position, sv_b.anx_position)
        self.assertEqual(sv_a.anx_time, sv_b.anx_time)
        self.assertEqual(sv_a.orbit_direction, sv_b.orbit_direction)
        self.assertEqual(sv_a.orbit_number, sv_b.orbit_number)
        self.assertEqual(sv_a.reference_time, sv_b.reference_time)
        self.assertEqual(sv_a.number_of_state_vectors, sv_b.number_of_state_vectors)
        self.assertEqual(sv_a.time_step, sv_b.time_step)
        self.assertEqual(sv_a.track_number, sv_b.track_number)

        self.assertEqual(sv_a.position_vector.shape, sv_b.position_vector.shape)
        for pos_a, pos_b in zip(sv_a.position_vector, sv_b.position_vector):
            for comp_a, comp_b in zip(pos_a, pos_b):
                self.assertEqual(comp_a, comp_b)

        self.assertEqual(sv_a.velocity_vector.shape, sv_b.velocity_vector.shape)
        for vel_a, vel_b in zip(sv_a.velocity_vector, sv_b.velocity_vector):
            for comp_a, comp_b in zip(vel_a, vel_b):
                self.assertEqual(comp_a, comp_b)

    def setUp(self):
        self.state_vectors_model = models.StateVectorDataType(
            p_sv_m=models.StateVectorDataType.PSvM(
                [
                    models.StateVectorDataType.PSvM.Val(1.2, n=1),
                    models.StateVectorDataType.PSvM.Val(2.3, n=2),
                    models.StateVectorDataType.PSvM.Val(3.4, n=3),
                    models.StateVectorDataType.PSvM.Val(4.5, n=4),
                    models.StateVectorDataType.PSvM.Val(5.6, n=5),
                    models.StateVectorDataType.PSvM.Val(6.7, n=6),
                ]
            ),
            v_sv_m_os=models.StateVectorDataType.VSvMOs(
                [
                    models.StateVectorDataType.VSvMOs.Val(11.2, n=1),
                    models.StateVectorDataType.VSvMOs.Val(12.3, n=2),
                    models.StateVectorDataType.VSvMOs.Val(13.4, n=3),
                    models.StateVectorDataType.VSvMOs.Val(14.5, n=4),
                    models.StateVectorDataType.VSvMOs.Val(15.6, n=5),
                    models.StateVectorDataType.VSvMOs.Val(16.7, n=6),
                ]
            ),
            orbit_number="NOT_AVAILABLE",
            track="NOT_AVAILABLE",
            orbit_direction=models.AscendingDescendingType.ASCENDING,
            t_ref_utc="01-JAN-2020 00:00:00.000000000000",
            dt_sv_s=models.StateVectorDataType.DtSvS(1.2, unit=models.Units.S),
            n_sv_n=models.StateVectorDataType.NSvN(2),
        )

        positions = np.zeros(shape=(2, 3))
        positions[0, :] = [1.2, 2.3, 3.4]
        positions[1, :] = [4.5, 5.6, 6.7]

        velocities = np.zeros(shape=(2, 3))
        velocities[0, :] = [11.2, 12.3, 13.4]
        velocities[1, :] = [14.5, 15.6, 16.7]

        self.state_vectors_metadata = metadata.StateVectors(
            position_vector=positions,
            velocity_vector=velocities,
            t_ref_utc=PreciseDateTime.from_numeric_datetime(year=2020),
            dt_sv_s=1.2,
        )

    def test_translate_state_vectors(self):
        self.assertEqual(
            translate_state_vectors_to_model(self.state_vectors_metadata),
            self.state_vectors_model,
        )
        self.assertEqualStateVectors(
            translate_state_vectors_from_model(self.state_vectors_model),
            self.state_vectors_metadata,
        )

    def test_translate_state_vectors_anx(self):
        self.state_vectors_metadata.set_anx_info(
            PreciseDateTime.from_numeric_datetime(year=2019), [1, 2, 3]
        )
        self.state_vectors_model.ascending_node_coords = (
            models.StateVectorDataType.AscendingNodeCoords(
                [
                    models.StateVectorDataType.AscendingNodeCoords.Val(1),
                    models.StateVectorDataType.AscendingNodeCoords.Val(2),
                    models.StateVectorDataType.AscendingNodeCoords.Val(3),
                ]
            )
        )
        self.state_vectors_model.ascending_node_time = (
            "01-JAN-2019 00:00:00.000000000000"
        )

        self.assertEqual(
            translate_state_vectors_to_model(self.state_vectors_metadata),
            self.state_vectors_model,
        )
        self.assertEqualStateVectors(
            translate_state_vectors_from_model(self.state_vectors_model),
            self.state_vectors_metadata,
        )

    def test_translate_state_vectors_auxiliary_info(self):
        self.state_vectors_metadata.orbit_number = 5
        self.state_vectors_metadata.track_number = 10
        self.state_vectors_model.orbit_number = "5"
        self.state_vectors_model.track = "10"

        self.assertEqual(
            translate_state_vectors_to_model(self.state_vectors_metadata),
            self.state_vectors_model,
        )
        self.assertEqualStateVectors(
            translate_state_vectors_from_model(self.state_vectors_model),
            self.state_vectors_metadata,
        )


class Poly2DTestCase(unittest.TestCase):
    def assertEqualPoly2D(self, poly_a: metadata._Poly2D, poly_b: metadata._Poly2D):
        self.assertEqual(poly_a.coefficients, poly_b.coefficients)
        self.assertEqual(poly_a.t_ref_az, poly_b.t_ref_az)
        self.assertEqual(poly_a.t_ref_rg, poly_b.t_ref_rg)

    def test_translate_polynomial(self):
        poly_metadata = metadata._Poly2D(
            i_ref_az=PreciseDateTime.from_numeric_datetime(year=2020),
            i_ref_rg=0.005,
            i_coefficients=[1, 2, 3, 4, 5, 6, 7],
        )
        poly_model = models.PolyType(
            pol=models.PolyType.Pol(
                [
                    models.PolyType.Pol.Val(1, n=1, unit=models.Units.VALUE),
                    models.PolyType.Pol.Val(2, n=2, unit=models.Units.VALUE),
                    models.PolyType.Pol.Val(3, n=3, unit=models.Units.VALUE),
                    models.PolyType.Pol.Val(4, n=4, unit=models.Units.VALUE),
                    models.PolyType.Pol.Val(5, n=5, unit=models.Units.VALUE),
                    models.PolyType.Pol.Val(6, n=6, unit=models.Units.VALUE),
                    models.PolyType.Pol.Val(7, n=7, unit=models.Units.VALUE),
                ]
            ),
            trg0_s=models.PolyType.Trg0S(0.005, unit=models.Units.S),
            taz0_utc=models.PolyType.Taz0Utc(
                "01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
            ),
        )

        self.assertEqualPoly2D(
            translate_polynomial_from_model(poly_model), poly_metadata
        )
        self.assertEqual(translate_polynomial_to_model(poly_metadata), poly_model)


class Poly2DListTestCase(unittest.TestCase):
    def assertEqualPolyList(
        self, poly_list_a: metadata._Poly2DVector, poly_list_b: metadata._Poly2DVector
    ):
        self.assertEqual(type(poly_list_a), type(poly_list_a))

        for poly_a, poly_b in zip(poly_list_a, poly_list_b):
            self.assertEqual(type(poly_a), type(poly_b))
            self.assertEqual(poly_a.coefficients, poly_b.coefficients)
            self.assertEqual(poly_a.t_ref_az, poly_b.t_ref_az)
            self.assertEqual(poly_a.t_ref_rg, poly_b.t_ref_rg)

    def test_translate_doppler_centroid(self):
        doppler_centroid_metadata = metadata.DopplerCentroidVector(
            [
                metadata.DopplerCentroid(
                    i_ref_az=PreciseDateTime.from_numeric_datetime(year=2020),
                    i_ref_rg=0.005,
                    i_coefficients=[1, 2, 3, 4, 5, 6, 7],
                )
            ]
        )

        doppler_centroid_model = [
            models.PolyType(
                number=1,
                total=1,
                pol=models.PolyType.Pol(
                    [
                        models.PolyType.Pol.Val(1, n=1, unit=models.Units.HZ),
                        models.PolyType.Pol.Val(2, n=2, unit=models.Units.HZ_S),
                        models.PolyType.Pol.Val(3, n=3, unit=models.Units.HZ_S),
                        models.PolyType.Pol.Val(4, n=4, unit=models.Units.HZ_S2),
                        models.PolyType.Pol.Val(5, n=5, unit=models.Units.HZ_S2),
                        models.PolyType.Pol.Val(6, n=6, unit=models.Units.HZ_S3),
                        models.PolyType.Pol.Val(7, n=7, unit=models.Units.HZ_S4),
                    ]
                ),
                trg0_s=models.PolyType.Trg0S(0.005, unit=models.Units.S),
                taz0_utc=models.PolyType.Taz0Utc(
                    "01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
                ),
            )
        ]

        self.assertEqual(
            translate_polynomial_list_to_model(doppler_centroid_metadata),
            doppler_centroid_model,
        )
        self.assertEqualPolyList(
            translate_polynomial_list_from_model(
                doppler_centroid_model,
                specific_type=metadata.DopplerCentroidVector,
            ),
            doppler_centroid_metadata,
        )

    def test_translate_doppler_centroid_all_orders(self):
        doppler_centroid_metadata = metadata.DopplerCentroidVector(
            [
                metadata.DopplerCentroid(
                    i_ref_az=PreciseDateTime.from_numeric_datetime(year=2020),
                    i_ref_rg=0.005,
                    i_coefficients=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                )
            ]
        )

        doppler_centroid_model = [
            models.PolyType(
                number=1,
                total=1,
                pol=models.PolyType.Pol(
                    [
                        models.PolyType.Pol.Val(1, n=1, unit=models.Units.HZ),
                        models.PolyType.Pol.Val(2, n=2, unit=models.Units.HZ_S),
                        models.PolyType.Pol.Val(3, n=3, unit=models.Units.HZ_S),
                        models.PolyType.Pol.Val(4, n=4, unit=models.Units.HZ_S2),
                        models.PolyType.Pol.Val(5, n=5, unit=models.Units.HZ_S2),
                        models.PolyType.Pol.Val(6, n=6, unit=models.Units.HZ_S3),
                        models.PolyType.Pol.Val(7, n=7, unit=models.Units.HZ_S4),
                        models.PolyType.Pol.Val(8, n=8, unit=models.Units.HZ_S5),
                        models.PolyType.Pol.Val(9, n=9, unit=models.Units.HZ_S6),
                        models.PolyType.Pol.Val(10, n=10, unit=models.Units.HZ_S7),
                        models.PolyType.Pol.Val(11, n=11, unit=models.Units.HZ_S8),
                    ]
                ),
                trg0_s=models.PolyType.Trg0S(0.005, unit=models.Units.S),
                taz0_utc=models.PolyType.Taz0Utc(
                    "01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
                ),
            )
        ]

        self.assertEqual(
            translate_polynomial_list_to_model(doppler_centroid_metadata),
            doppler_centroid_model,
        )
        self.assertEqualPolyList(
            translate_polynomial_list_from_model(
                doppler_centroid_model,
                specific_type=metadata.DopplerCentroidVector,
            ),
            doppler_centroid_metadata,
        )

    def test_translate_doppler_rate(self):
        doppler_rate_metadata = metadata.DopplerRateVector(
            [
                metadata.DopplerRate(
                    i_ref_az=PreciseDateTime.from_numeric_datetime(year=2020),
                    i_ref_rg=0.005,
                    i_coefficients=[1, 2, 3, 4, 5, 6, 7],
                )
            ]
        )

        doppler_rate_model = [
            models.PolyType(
                number=1,
                total=1,
                pol=models.PolyType.Pol(
                    [
                        models.PolyType.Pol.Val(1, n=1, unit=models.Units.HZ_S),
                        models.PolyType.Pol.Val(2, n=2, unit=models.Units.HZ_S2),
                        models.PolyType.Pol.Val(3, n=3, unit=models.Units.HZ_S2),
                        models.PolyType.Pol.Val(4, n=4, unit=models.Units.HZ_S3),
                        models.PolyType.Pol.Val(5, n=5, unit=models.Units.HZ_S3),
                        models.PolyType.Pol.Val(6, n=6, unit=models.Units.HZ_S4),
                        models.PolyType.Pol.Val(7, n=7, unit=models.Units.HZ_S5),
                    ]
                ),
                trg0_s=models.PolyType.Trg0S(0.005, unit=models.Units.S),
                taz0_utc=models.PolyType.Taz0Utc(
                    "01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
                ),
            )
        ]

        self.assertEqual(
            translate_polynomial_list_to_model(doppler_rate_metadata),
            doppler_rate_model,
        )
        self.assertEqualPolyList(
            translate_polynomial_list_from_model(
                doppler_rate_model,
                specific_type=metadata.DopplerRateVector,
            ),
            doppler_rate_metadata,
        )

    def test_translate_doppler_rate_all_orders(self):
        doppler_rate_metadata = metadata.DopplerRateVector(
            [
                metadata.DopplerRate(
                    i_ref_az=PreciseDateTime.from_numeric_datetime(year=2020),
                    i_ref_rg=0.005,
                    i_coefficients=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                )
            ]
        )

        doppler_rate_model = [
            models.PolyType(
                number=1,
                total=1,
                pol=models.PolyType.Pol(
                    [
                        models.PolyType.Pol.Val(1, n=1, unit=models.Units.HZ_S),
                        models.PolyType.Pol.Val(2, n=2, unit=models.Units.HZ_S2),
                        models.PolyType.Pol.Val(3, n=3, unit=models.Units.HZ_S2),
                        models.PolyType.Pol.Val(4, n=4, unit=models.Units.HZ_S3),
                        models.PolyType.Pol.Val(5, n=5, unit=models.Units.HZ_S3),
                        models.PolyType.Pol.Val(6, n=6, unit=models.Units.HZ_S4),
                        models.PolyType.Pol.Val(7, n=7, unit=models.Units.HZ_S5),
                        models.PolyType.Pol.Val(8, n=8, unit=models.Units.HZ_S6),
                        models.PolyType.Pol.Val(9, n=9, unit=models.Units.HZ_S7),
                        models.PolyType.Pol.Val(10, n=10, unit=models.Units.HZ_S8),
                        models.PolyType.Pol.Val(11, n=11, unit=models.Units.HZ_S9),
                    ]
                ),
                trg0_s=models.PolyType.Trg0S(0.005, unit=models.Units.S),
                taz0_utc=models.PolyType.Taz0Utc(
                    "01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
                ),
            )
        ]

        self.assertEqual(
            translate_polynomial_list_to_model(doppler_rate_metadata),
            doppler_rate_model,
        )
        self.assertEqualPolyList(
            translate_polynomial_list_from_model(
                doppler_rate_model,
                specific_type=metadata.DopplerRateVector,
            ),
            doppler_rate_metadata,
        )


class CoregPolyTestCase(unittest.TestCase):
    def assertEqualPoly2D(self, poly_a: metadata.CoregPoly, poly_b: metadata.CoregPoly):
        self.assertEqual(
            poly_a.azimuth_poly.coefficients, poly_b.azimuth_poly.coefficients
        )
        self.assertEqual(poly_a.range_poly.coefficients, poly_b.range_poly.coefficients)
        self.assertEqual(poly_a.ref_azimuth_time, poly_b.ref_azimuth_time)
        self.assertEqual(poly_a.ref_range_time, poly_b.ref_range_time)

    def test_translate_polynomial(self):
        poly_metadata = metadata.CoregPoly(
            i_ref_az=PreciseDateTime.from_numeric_datetime(year=2020),
            i_ref_rg=0.005,
            i_coefficients_az=[1, 2, 3, 4, 5, 6, 7],
            i_coefficients_rg=[11, 12, 13, 14, 15, 16, 17],
        )
        poly_model = models.PolyCoregType(
            pol_az=models.PolyCoregType.PolAz(
                [
                    models.PolyCoregType.PolAz.Val(1, n=1),
                    models.PolyCoregType.PolAz.Val(2, n=2),
                    models.PolyCoregType.PolAz.Val(3, n=3),
                    models.PolyCoregType.PolAz.Val(4, n=4),
                    models.PolyCoregType.PolAz.Val(5, n=5),
                    models.PolyCoregType.PolAz.Val(6, n=6),
                    models.PolyCoregType.PolAz.Val(7, n=7),
                ]
            ),
            pol_rg=models.PolyCoregType.PolRg(
                [
                    models.PolyCoregType.PolRg.Val(11, n=1),
                    models.PolyCoregType.PolRg.Val(12, n=2),
                    models.PolyCoregType.PolRg.Val(13, n=3),
                    models.PolyCoregType.PolRg.Val(14, n=4),
                    models.PolyCoregType.PolRg.Val(15, n=5),
                    models.PolyCoregType.PolRg.Val(16, n=6),
                    models.PolyCoregType.PolRg.Val(17, n=7),
                ]
            ),
            trg0_s=models.PolyCoregType.Trg0S(0.005, unit=models.Units.S),
            taz0_utc=models.PolyCoregType.Taz0Utc(
                "01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
            ),
        )

        self.assertEqualPoly2D(
            translate_coreg_polynomial_from_model(poly_model), poly_metadata
        )
        self.assertEqual(translate_coreg_polynomial_to_model(poly_metadata), poly_model)


class CoregPolyListTestCase(unittest.TestCase):
    def assertEqualPolyList(
        self,
        poly_list_a: metadata.CoregPolyVector,
        poly_list_b: metadata.CoregPolyVector,
    ):
        self.assertEqual(type(poly_list_a), type(poly_list_a))

        for poly_a, poly_b in zip(poly_list_a, poly_list_b):
            self.assertEqual(type(poly_a), type(poly_b))
            poly_a: metadata.CoregPoly
            poly_b: metadata.CoregPoly
            self.assertEqual(
                poly_a.azimuth_poly.coefficients, poly_b.azimuth_poly.coefficients
            )
            self.assertEqual(
                poly_a.range_poly.coefficients, poly_b.range_poly.coefficients
            )
            self.assertEqual(poly_a.ref_azimuth_time, poly_b.ref_azimuth_time)
            self.assertEqual(poly_a.ref_range_time, poly_b.ref_range_time)

    def test_translate_coreg_poly(self):
        coreg_poly_metadata = metadata.CoregPolyVector(
            [
                metadata.CoregPoly(
                    i_ref_az=PreciseDateTime.from_numeric_datetime(year=2020),
                    i_ref_rg=0.005,
                    i_coefficients_az=[1, 2, 3, 4, 5, 6, 7],
                    i_coefficients_rg=[11, 12, 13, 14, 15, 16, 17],
                )
            ]
        )

        coreg_poly_model = [
            models.PolyCoregType(
                number=1,
                total=1,
                pol_az=models.PolyCoregType.PolAz(
                    [
                        models.PolyCoregType.PolAz.Val(1, n=1),
                        models.PolyCoregType.PolAz.Val(2, n=2),
                        models.PolyCoregType.PolAz.Val(3, n=3),
                        models.PolyCoregType.PolAz.Val(4, n=4),
                        models.PolyCoregType.PolAz.Val(5, n=5),
                        models.PolyCoregType.PolAz.Val(6, n=6),
                        models.PolyCoregType.PolAz.Val(7, n=7),
                    ]
                ),
                pol_rg=models.PolyCoregType.PolRg(
                    [
                        models.PolyCoregType.PolRg.Val(11, n=1),
                        models.PolyCoregType.PolRg.Val(12, n=2),
                        models.PolyCoregType.PolRg.Val(13, n=3),
                        models.PolyCoregType.PolRg.Val(14, n=4),
                        models.PolyCoregType.PolRg.Val(15, n=5),
                        models.PolyCoregType.PolRg.Val(16, n=6),
                        models.PolyCoregType.PolRg.Val(17, n=7),
                    ]
                ),
                trg0_s=models.PolyCoregType.Trg0S(0.005, unit=models.Units.S),
                taz0_utc=models.PolyCoregType.Taz0Utc(
                    "01-JAN-2020 00:00:00.000000000000", unit=models.Units.UTC
                ),
            )
        ]

        self.assertEqual(
            translate_coreg_polynomial_list_to_model(coreg_poly_metadata),
            coreg_poly_model,
        )
        self.assertEqualPolyList(
            translate_coreg_polynomial_list_from_model(coreg_poly_model),
            coreg_poly_metadata,
        )


class DataStatisticsTestCase(unittest.TestCase):
    def assertEqualDataStatistics(
        self, stat_a: metadata.DataStatistics, stat_b: metadata.DataStatistics
    ):
        self.assertEqual(stat_a.num_samples, stat_b.num_samples)
        self.assertEqual(stat_a.max_i, stat_b.max_i)
        self.assertEqual(stat_a.max_q, stat_b.max_q)
        self.assertEqual(stat_a.min_i, stat_b.min_i)
        self.assertEqual(stat_a.min_q, stat_b.min_q)
        self.assertEqual(stat_a.sum_i, stat_b.sum_i)
        self.assertEqual(stat_a.sum_q, stat_b.sum_q)
        self.assertEqual(stat_a.sum_2_i, stat_b.sum_2_i)
        self.assertEqual(stat_a.sum_2_q, stat_b.sum_2_q)
        self.assertEqual(stat_a.std_dev_i, stat_b.std_dev_i)
        self.assertEqual(stat_a.std_dev_q, stat_b.std_dev_q)
        self.assertEqual(
            stat_a.get_number_of_data_block_statistic(),
            stat_b.get_number_of_data_block_statistic(),
        )
        for index in range(stat_a.get_number_of_data_block_statistic()):
            block_a = stat_a.get_data_block_statistic(index)
            block_b = stat_b.get_data_block_statistic(index)
            self.assertEqual(block_a.num_samples, block_b.num_samples)
            self.assertEqual(block_a.max_i, block_b.max_i)
            self.assertEqual(block_a.max_q, block_b.max_q)
            self.assertEqual(block_a.min_i, block_b.min_i)
            self.assertEqual(block_a.min_q, block_b.min_q)
            self.assertEqual(block_a.sum_i, block_b.sum_i)
            self.assertEqual(block_a.sum_q, block_b.sum_q)
            self.assertEqual(block_a.sum_2_i, block_b.sum_2_i)
            self.assertEqual(block_a.sum_2_q, block_b.sum_2_q)

    def setUp(self):
        self.stat_metadata = metadata.DataStatistics(
            i_num_samples=5,
            i_max_i=2.1,
            i_min_i=-2.0,
            i_max_q=8.5,
            i_min_q=-9.6,
            i_sum_i=20.2,
            i_sum_q=-5.2,
            i_sum_2_i=50.0,
            i_sum_2_q=81,
            i_std_dev_i=5.0,
            i_std_dev_q=6.4,
        )

        self.stat_model = metadata_models.DataStatisticsType(
            num_samples=metadata_models.DataStatisticsType.NumSamples(5),
            max_i=metadata_models.DataStatisticsType.MaxI(2.1),
            min_i=metadata_models.DataStatisticsType.MinI(-2.0),
            max_q=metadata_models.DataStatisticsType.MaxQ(8.5),
            min_q=metadata_models.DataStatisticsType.MinQ(-9.6),
            sum_i=metadata_models.DataStatisticsType.SumI(20.2),
            sum_q=metadata_models.DataStatisticsType.SumQ(-5.2),
            sum2_i=metadata_models.DataStatisticsType.Sum2I(50.0),
            sum2_q=metadata_models.DataStatisticsType.Sum2Q(81),
            std_dev_i=metadata_models.DataStatisticsType.StdDevI(5.0),
            std_dev_q=metadata_models.DataStatisticsType.StdDevQ(6.4),
        )

    def test_translate_data_statistics(self):
        self.assertEqualDataStatistics(
            translate_data_statistics_from_model(self.stat_model), self.stat_metadata
        )
        self.assertEqual(
            translate_data_statistics_to_model(self.stat_metadata), self.stat_model
        )


class SensorNamesTestCase(unittest.TestCase):
    def test_translate_sensor_name(self):
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.NOT_SET), "NOT SET"
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.ASAR), "ASAR"
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.PALSAR), "PALSAR"
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.ERS1), "ERS1"
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.ERS2), "ERS2"
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.RADARSAT),
            "RADARSAT",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.TERRASARX),
            "TERRASARX",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.SENTINEL1),
            "SENTINEL1",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.SENTINEL1_A),
            "SENTINEL1A",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.SENTINEL1_B),
            "SENTINEL1B",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.SENTINEL1_C),
            "SENTINEL1C",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.SENTINEL1_D),
            "SENTINEL1D",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.SAOCOM), "SAOCOM"
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.SAOCOM_1_A),
            "SAOCOM-1A",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.SAOCOM_1_B),
            "SAOCOM-1B",
        )
        self.assertEqual(
            translate_sensor_names_from_model(models.SensorNamesType.UAVSAR), "UAVSAR"
        )

        self.assertEqual(
            translate_sensor_names_to_model("NOT SET"), models.SensorNamesType.NOT_SET
        )
        self.assertEqual(
            translate_sensor_names_to_model("ASAR"), models.SensorNamesType.ASAR
        )
        self.assertEqual(
            translate_sensor_names_to_model("PALSAR"), models.SensorNamesType.PALSAR
        )
        self.assertEqual(
            translate_sensor_names_to_model("ERS1"), models.SensorNamesType.ERS1
        )
        self.assertEqual(
            translate_sensor_names_to_model("ERS2"), models.SensorNamesType.ERS2
        )
        self.assertEqual(
            translate_sensor_names_to_model("RADARSAT"), models.SensorNamesType.RADARSAT
        )
        self.assertEqual(
            translate_sensor_names_to_model("TERRASARX"),
            models.SensorNamesType.TERRASARX,
        )
        self.assertEqual(
            translate_sensor_names_to_model("SENTINEL1"),
            models.SensorNamesType.SENTINEL1,
        )
        self.assertEqual(
            translate_sensor_names_to_model("SENTINEL1A"),
            models.SensorNamesType.SENTINEL1_A,
        )
        self.assertEqual(
            translate_sensor_names_to_model("SENTINEL1B"),
            models.SensorNamesType.SENTINEL1_B,
        )
        self.assertEqual(
            translate_sensor_names_to_model("SENTINEL1C"),
            models.SensorNamesType.SENTINEL1_C,
        )
        self.assertEqual(
            translate_sensor_names_to_model("SENTINEL1D"),
            models.SensorNamesType.SENTINEL1_D,
        )
        self.assertEqual(
            translate_sensor_names_to_model("SAOCOM"), models.SensorNamesType.SAOCOM
        )
        self.assertEqual(
            translate_sensor_names_to_model("SAOCOM-1A"),
            models.SensorNamesType.SAOCOM_1_A,
        )
        self.assertEqual(
            translate_sensor_names_to_model("SAOCOM-1B"),
            models.SensorNamesType.SAOCOM_1_B,
        )
        self.assertEqual(
            translate_sensor_names_to_model("UAVSAR"), models.SensorNamesType.UAVSAR
        )


class AntennaInfoTestCase(unittest.TestCase):
    def assertEqualAntennaInfo(
        self, info_a: metadata.AntennaInfo, info_b: metadata.AntennaInfo
    ):
        self.assertEqual(info_a.sensor_name, info_b.sensor_name)
        self.assertEqual(info_a.polarization, info_b.polarization)
        self.assertEqual(info_a.acquisition_mode, info_b.acquisition_mode)
        self.assertEqual(info_a.acquisition_beam, info_b.acquisition_beam)
        self.assertEqual(info_a.lines_per_pattern, info_b.lines_per_pattern)

    def test_translate_antenna_info(self):
        info_model = models.AntennaInfoType(
            beam_name="S2",
            sensor_name=models.SensorNamesType.NOT_SET,
            acquisition_mode=metadata_models.AcquisitionModeType.STRIPMAP,
            polarization=models.PolarizationType.H_H,
        )
        info_metadata = metadata.AntennaInfo(
            "NOT SET", metadata.EPolarization.hh, "STRIPMAP", "S2"
        )

        self.assertEqual(translate_antenna_info_to_model(info_metadata), info_model)
        self.assertEqualAntennaInfo(
            translate_antenna_info_from_model(info_model), info_metadata
        )

        info_model.lines_per_pattern = 4
        info_metadata.lines_per_pattern = 4

        self.assertEqual(translate_antenna_info_to_model(info_metadata), info_model)
        self.assertEqualAntennaInfo(
            translate_antenna_info_from_model(info_model), info_metadata
        )


class PulseDirectionTestCase(unittest.TestCase):
    def test_translate_pulse_direction(self):
        self.assertEqual(
            translate_pulse_direction_to_model(metadata.EPulseDirection.down),
            metadata_models.PulseTypeDirection.DOWN,
        )
        self.assertEqual(
            translate_pulse_direction_to_model(metadata.EPulseDirection.up),
            metadata_models.PulseTypeDirection.UP,
        )
        self.assertEqual(
            translate_pulse_direction_from_model(
                metadata_models.PulseTypeDirection.DOWN
            ),
            metadata.EPulseDirection.down,
        )
        self.assertEqual(
            translate_pulse_direction_from_model(metadata_models.PulseTypeDirection.UP),
            metadata.EPulseDirection.up,
        )


class PulseTestCase(unittest.TestCase):
    def assertEqualPulse(self, pulse_a: metadata.Pulse, pulse_b: metadata.Pulse):
        self.assertEqual(pulse_a.pulse_length, pulse_b.pulse_length)
        self.assertEqual(pulse_a.pulse_length_unit, pulse_b.pulse_length_unit)
        self.assertEqual(pulse_a.bandwidth, pulse_b.bandwidth)
        self.assertEqual(pulse_a.bandwidth_unit, pulse_b.bandwidth_unit)
        self.assertEqual(pulse_a.pulse_energy, pulse_b.pulse_energy)
        self.assertEqual(pulse_a.pulse_energy_unit, pulse_b.pulse_energy_unit)
        self.assertEqual(pulse_a.pulse_sampling_rate, pulse_b.pulse_sampling_rate)
        self.assertEqual(
            pulse_a.pulse_sampling_rate_unit, pulse_b.pulse_sampling_rate_unit
        )
        self.assertEqual(pulse_a.pulse_start_frequency, pulse_b.pulse_start_frequency)
        self.assertEqual(
            pulse_a.pulse_start_frequency_unit, pulse_b.pulse_start_frequency_unit
        )
        self.assertEqual(pulse_a.pulse_start_phase, pulse_b.pulse_start_phase)
        self.assertEqual(pulse_a.pulse_start_phase_unit, pulse_b.pulse_start_phase_unit)
        self.assertEqual(pulse_a.pulse_direction, pulse_b.pulse_direction)

    def test_translate_pulse(self):
        pulse_metadata = metadata.Pulse(
            i_pulse_length=0.005,
            i_bandwidth=100000.5,
            i_pulse_sampling_rate=120000.5,
        )

        pulse_model = models.PulseType(
            pulse_length=models.DoubleWithUnit(0.005, models.Units.S),
            bandwidth=models.DoubleWithUnit(100000.5, models.Units.HZ),
            pulse_sampling_rate=models.DoubleWithUnit(120000.5, models.Units.HZ),
        )

        self.assertEqual(translate_pulse_to_model(pulse_metadata), pulse_model)
        self.assertEqualPulse(translate_pulse_from_model(pulse_model), pulse_metadata)

    def test_translate_pulse_additional_data(self):
        pulse_metadata = metadata.Pulse(
            i_pulse_length=0.005,
            i_bandwidth=100000.5,
            i_pulse_sampling_rate=120000.6,
            i_pulse_energy=56,
            i_pulse_start_frequency=-8,
            i_pulse_start_phase=-0.5,
            i_pulse_direction=metadata.EPulseDirection.down,
        )

        pulse_model = models.PulseType(
            direction=models.PulseTypeDirection.DOWN,
            pulse_length=models.DoubleWithUnit(0.005, models.Units.S),
            bandwidth=models.DoubleWithUnit(100000.5, models.Units.HZ),
            pulse_energy=models.DoubleWithUnit(56, models.Units.J),
            pulse_sampling_rate=models.DoubleWithUnit(120000.6, models.Units.HZ),
            pulse_start_frequency=models.DoubleWithUnit(-8, models.Units.HZ),
            pulse_start_phase=models.DoubleWithUnit(-0.5, models.Units.RAD),
        )

        self.assertEqual(translate_pulse_to_model(pulse_metadata), pulse_model)
        self.assertEqualPulse(translate_pulse_from_model(pulse_model), pulse_metadata)


class MetadataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata_obj = metadata.MetaData(description="Description")

        mdc = metadata.MetaDataChannel()
        raster_info = models.RasterInfoType(
            file_name="filename.tiff",
            lines=11,
            samples=3,
            header_offset_bytes=50,
            row_prefix_bytes=4,
            byte_order=models.Endianity.BIGENDIAN,
            cell_type=models.CellTypeVerboseType.INT16,
            lines_step=translate_double_with_unit_to_model(0.5, "Hz"),
            samples_step=translate_double_with_unit_to_model(8.5, "m"),
            invalid_value=None,
            lines_start=translate_str_with_unit_to_model(
                PreciseDateTime.from_numeric_datetime(year=2020), "Utc"
            ),
            samples_start=translate_str_with_unit_to_model(3.4, "m"),
            raster_format=None,
        )

        mdc.insert_element(translate_raster_info_from_model(raster_info))
        self.metadata_obj.append_channel(mdc)

        self.model_obj = metadata_models.AresysXmlDoc(1, 2.1, "Description")
        channel_model = metadata_models.AresysXmlDoc.Channel(number=1, total=1)
        channel_model.raster_info = raster_info
        self.model_obj.channel.append(channel_model)

    def test_translate_metadata(self):
        self.assertEqual(translate_metadata_to_model(self.metadata_obj), self.model_obj)
        self.assertEqual(
            translate_metadata_to_model(translate_metadata_from_model(self.model_obj)),
            self.model_obj,
        )

    def test_translate_metadata_sub_channels(self):
        channel_two_model = metadata_models.AresysXmlDoc.Channel(number=2, total=2)
        channel_two_model.pulse = translate_pulse_to_model(
            metadata.Pulse(i_pulse_length=0.5, i_bandwidth=80, i_pulse_sampling_rate=90)
        )
        self.model_obj.channel[0].total = 2
        self.model_obj.channel.append(channel_two_model)
        self.model_obj.number_of_channels = 2

        channel_two_metadata = metadata.MetaDataChannel()
        channel_two_metadata.insert_element(
            metadata.Pulse(i_pulse_length=0.5, i_bandwidth=80, i_pulse_sampling_rate=90)
        )
        self.metadata_obj.append_channel(channel_two_metadata)

        self.assertEqual(translate_metadata_to_model(self.metadata_obj), self.model_obj)
        self.assertEqual(
            translate_metadata_to_model(translate_metadata_from_model(self.model_obj)),
            self.model_obj,
        )


if __name__ == "__main__":
    unittest.main()
