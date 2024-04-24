# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for io/io_support functionalities"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import arepytools.io.io_support as support
from arepytools.io import metadata as mtd
from arepytools.io.parsing.translate import translate_metadata_to_model
from arepytools.timing.precisedatetime import PreciseDateTime

METADATA = """<?xml version="1.0" encoding="utf-8"?>
<AresysXmlDoc>
  <NumberOfChannels>1</NumberOfChannels>
  <VersionNumber>2.1</VersionNumber>
  <Description/>
  <Channel Number="1" Total="1">
    <RasterInfo>
      <FileName>GRD_0001</FileName>
      <Lines>8951</Lines>
      <Samples>2215</Samples>
      <HeaderOffsetBytes>150</HeaderOffsetBytes>
      <RowPrefixBytes>20</RowPrefixBytes>
      <ByteOrder>LITTLEENDIAN</ByteOrder>
      <CellType>FLOAT32</CellType>
      <LinesStep unit="s">0.00325489654798287</LinesStep>
      <SamplesStep unit="m">25.0</SamplesStep>
      <LinesStart unit="Utc">11-JAN-2017 05:06:05.420354672133</LinesStart>
      <SamplesStart unit="m">0.0</SamplesStart>
    </RasterInfo>
  </Channel>
</AresysXmlDoc>
"""


class MetadataInputOutputTestCase(unittest.TestCase):
    def assertEqualMetadata(self, metadata_a: mtd.MetaData, metadata_b: mtd.MetaData):
        model_a = translate_metadata_to_model(metadata_a)
        model_b = translate_metadata_to_model(metadata_b)
        self.assertEqual(model_a, model_b)

    def setUp(self) -> None:
        self.maxDiff = None

        self.metadata_str = METADATA
        self.metadata_obj = mtd.MetaData(description="")

        channel = mtd.MetaDataChannel()
        channel.number = 1
        channel.total = 1

        raster_info = mtd.RasterInfo(
            lines=8951,
            samples=2215,
            celltype="FLOAT32",
            filename="GRD_0001",
            header_offset_bytes=150,
            row_prefix_bytes=20,
            byteorder="LITTLEENDIAN",
            invalid_value=None,
            format_type=None,
        )
        raster_info.set_lines_axis(
            PreciseDateTime.from_utc_string("11-JAN-2017 05:06:05.420354672133"),
            "Utc",
            0.00325489654798287,
            "s",
        )
        raster_info.set_samples_axis(0.0, "m", 25.0, "m")
        channel.insert_element(raster_info)

        self.metadata_obj.append_channel(channel)

    def test_read(self):
        with TemporaryDirectory() as temp_dir:
            metadata = Path(temp_dir).joinpath("metadata.xml")
            metadata.write_text(self.metadata_str)
            self.assertEqualMetadata(support.read_metadata(metadata), self.metadata_obj)

    def test_write(self):
        with TemporaryDirectory() as temp_dir:
            metadata = Path(temp_dir).joinpath("metadata.xml")
            support.write_metadata(self.metadata_obj, metadata)
            self.assertEqual(metadata.read_text(encoding="utf-8"), self.metadata_str)

    def test_create(self):
        """Testing create_new_metadata"""
        meta = support.create_new_metadata(10, "test")
        self.assertEqual("test", meta.description)
        self.assertEqual(10, meta.get_number_of_channels())


class SupportFunctionsTest(unittest.TestCase):
    """Testing io_support main functions"""

    def setUp(self) -> None:
        self.lines = 600
        self.samples = 350
        self.header_offset = 150
        self.row_prefix = 20
        self.raster_info = mtd.RasterInfo(
            lines=self.lines,
            samples=self.samples,
            celltype="FLOAT32",
            filename="GRD_0001",
            header_offset_bytes=self.header_offset,
            row_prefix_bytes=self.row_prefix,
            byteorder="LITTLEENDIAN",
            invalid_value=None,
            format_type=None,
        )
        self.data = np.ones([self.lines, self.samples])

    def test_write_raster_with_raster_info(self) -> None:
        """Testing write_raster_with_raster_info function"""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            file = path.joinpath(self.raster_info.file_name)
            self.assertFalse(file.exists())

            support.write_raster_with_raster_info(
                raster_file=file, data=self.data, raster_info=self.raster_info
            )
            self.assertTrue(file.exists())
            self.assertTrue(file.is_file())

    def test_read_raster_with_raster_info(self) -> None:
        """Testing read_raster_with_raster_info function"""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            file = path.joinpath(self.raster_info.file_name)
            self.assertFalse(file.exists())

            support.write_raster_with_raster_info(
                raster_file=file, data=self.data, raster_info=self.raster_info
            )
            self.assertTrue(file.exists())
            self.assertTrue(file.is_file())

            read = support.read_raster_with_raster_info(
                raster_file=file, raster_info=self.raster_info
            )

            np.testing.assert_array_equal(read, self.data)

    def test_read_binary_header_with_raster_info(self) -> None:
        """Testing read_binary_header_with_raster_info function"""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            file = path.joinpath(self.raster_info.file_name)
            self.assertFalse(file.exists())

            support.write_raster_with_raster_info(
                raster_file=file, data=self.data, raster_info=self.raster_info
            )
            self.assertTrue(file.exists())
            self.assertTrue(file.is_file())

            read = support.read_binary_header_with_raster_info(
                raster_file=file, raster_info=self.raster_info
            )

            self.assertEqual(bytes(self.header_offset), read)

    def test_write_binary_header_with_raster_info(self) -> None:
        """Testing write_binary_header_with_raster_info function"""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            file = path.joinpath(self.raster_info.file_name)
            self.assertFalse(file.exists())

            support.write_raster_with_raster_info(
                raster_file=file, data=self.data, raster_info=self.raster_info
            )
            self.assertTrue(file.exists())
            self.assertTrue(file.is_file())

            support.write_binary_header_with_raster_info(
                raster_file=file,
                header=bytes(self.header_offset),
                raster_info=self.raster_info,
            )

    def test_write_binary_header_with_raster_info_error(self) -> None:
        """Testing write_binary_header_with_raster_info function, raising error"""
        with self.assertRaises(support.InvalidHeaderOffset):
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                file = path.joinpath(self.raster_info.file_name)
                self.assertFalse(file.exists())

                support.write_raster_with_raster_info(
                    raster_file=file, data=self.data, raster_info=self.raster_info
                )
                self.assertTrue(file.exists())
                self.assertTrue(file.is_file())

                support.write_binary_header_with_raster_info(
                    raster_file=file,
                    header=bytes(self.header_offset + 1),
                    raster_info=self.raster_info,
                )

    def test_read_row_prefix_with_raster_info(self) -> None:
        """Testing read_row_prefix_with_raster_info function"""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            file = path.joinpath(self.raster_info.file_name)
            self.assertFalse(file.exists())

            support.write_raster_with_raster_info(
                raster_file=file, data=self.data, raster_info=self.raster_info
            )
            self.assertTrue(file.exists())
            self.assertTrue(file.is_file())

            read = support.read_row_prefix_with_raster_info(
                raster_file=file, line_index=3, raster_info=self.raster_info
            )

            self.assertEqual(bytes(self.row_prefix), read)

    def test_write_row_prefix_with_raster_info(self) -> None:
        """Testing write_row_prefix_with_raster_info function"""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            file = path.joinpath(self.raster_info.file_name)
            self.assertFalse(file.exists())

            support.write_raster_with_raster_info(
                raster_file=file, data=self.data, raster_info=self.raster_info
            )
            self.assertTrue(file.exists())
            self.assertTrue(file.is_file())

            support.write_row_prefix_with_raster_info(
                raster_file=file,
                line_index=3,
                row_prefix=bytes(self.row_prefix),
                raster_info=self.raster_info,
            )

    def test_write_row_prefix_with_raster_info_error(self) -> None:
        """Testing write_row_prefix_with_raster_info function, raising error"""
        with self.assertRaises(support.InvalidRowPrefixSize):
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                file = path.joinpath(self.raster_info.file_name)
                self.assertFalse(file.exists())

                support.write_raster_with_raster_info(
                    raster_file=file, data=self.data, raster_info=self.raster_info
                )
                self.assertTrue(file.exists())
                self.assertTrue(file.is_file())

                support.write_row_prefix_with_raster_info(
                    raster_file=file,
                    line_index=3,
                    row_prefix=bytes(self.row_prefix + 1),
                    raster_info=self.raster_info,
                )


if __name__ == "__main__":
    unittest.main()
