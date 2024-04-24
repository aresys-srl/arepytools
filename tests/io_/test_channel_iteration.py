# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import GeneratorType
from unittest.mock import Mock

from arepytools.io import channel_iteration, create_product_folder, write_metadata
from arepytools.io.metadata import MetaData

swaths = ["S1", "S2", "S3"]
pols = ["H/H", "V/V", "V/H", "H/V"]

_xml = """
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<AresysXmlDoc xmlns:at="aresysTypes">
  <NumberOfChannels>1</NumberOfChannels>
  <VersionNumber>2.1</VersionNumber>
  <Description/>
  <Channel Number="1" Total="1">
    <RasterInfo>
      <FileName>iSLC_0001</FileName>
      <Lines>42878</Lines>
      <Samples>1210</Samples>
      <HeaderOffsetBytes>0</HeaderOffsetBytes>
      <RowPrefixBytes>0</RowPrefixBytes>
      <ByteOrder>LITTLEENDIAN</ByteOrder>
      <CellType>FLOAT_COMPLEX</CellType>
      <LinesStep unit="s">0.000679478160138531</LinesStep>
      <SamplesStep unit="s">1.32183907894041e-07</SamplesStep>
      <LinesStart unit="Utc">01-JAN-2017 06:03:05.415896855135</LinesStart>
      <SamplesStart unit="s">0.004821408040426434886</SamplesStart>
    </RasterInfo>
    <SwathInfo>
      <Swath>XXXX</Swath>
      <SwathAcquisitionOrder>0</SwathAcquisitionOrder>
      <Polarization>YYYY</Polarization>
      <Rank>7</Rank>
      <RangeDelayBias>0</RangeDelayBias>
      <AcquisitionStartTime>01-JAN-2017 06:03:05.418499000000</AcquisitionStartTime>
      <AzimuthSteeringRateReferenceTime unit="s">0</AzimuthSteeringRateReferenceTime>
      <AzimuthSteeringRatePol>
        <val N="1">0</val>
        <val N="2">0</val>
        <val N="3">0</val>
      </AzimuthSteeringRatePol>
      <AcquisitionPRF>1471.71764843203</AcquisitionPRF>
      <EchoesPerBurst>42878</EchoesPerBurst>
      <RxGain>1</RxGain>
    </SwathInfo>
  </Channel>
</AresysXmlDoc>
"""


class ChannelTestCase(unittest.TestCase):
    """Testing channel_iteration functions"""

    def setUp(self) -> None:
        self.pol_xml = [
            _xml.replace(
                "<Polarization>YYYY</Polarization>", f"<Polarization>{p}</Polarization>"
            )
            for p in pols
        ]
        self.swt_xml = [
            _xml.replace("<Swath>XXXX</Swath>", f"<Swath>{s}</Swath>") for s in swaths
        ]
        self.filter_pol = channel_iteration.SwathIDFilter(polarization=pols[1])
        self.filter_swt = channel_iteration.SwathIDFilter(swath=swaths[1])
        self.filter_both = channel_iteration.SwathIDFilter(
            polarization=pols[1], swath=swaths[1]
        )

    def test_iter_channels_generator_no_filter(self) -> None:
        """Testing iter_channels_generator function, no filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")
            out = channel_iteration.iter_channels_generator(product=pf)

            # check results
            self.assertIsInstance(out, GeneratorType)
            for idx, item in enumerate(out):
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[idx])

    def test_iter_channels_generator_filter(self) -> None:
        """Testing iter_channels_generator function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")
            out = channel_iteration.iter_channels_generator(
                product=pf, filter_func=self.filter_pol
            )

            # check results
            self.assertIsInstance(out, GeneratorType)
            for item in out:
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[1])

    def test_iter_channels_generator_filter_1(self) -> None:
        """Testing iter_channels_generator function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")
            out = channel_iteration.iter_channels_generator(
                product=pf, filter_func=self.filter_pol
            )

            # check results
            self.assertIsInstance(out, GeneratorType)
            for item in out:
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[1])

    def test_iter_channels_generator_filter_2(self) -> None:
        """Testing iter_channels_generator function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                content = content.replace(
                    "<Swath>XXXX</Swath>", f"<Swath>{swaths[1]}</Swath>"
                )
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")

            out = channel_iteration.iter_channels_generator(
                product=pf, filter_func=self.filter_swt
            )

            # check results
            self.assertIsInstance(out, GeneratorType)
            for item in out:
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().swath, swaths[1])

    def test_iter_channels_generator_filter_3(self) -> None:
        """Testing iter_channels_generator function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                content = content.replace(
                    "<Swath>XXXX</Swath>", f"<Swath>{swaths[1]}</Swath>"
                )
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")

            out = channel_iteration.iter_channels_generator(
                product=pf, filter_func=self.filter_both
            )

            # check results
            self.assertIsInstance(out, GeneratorType)
            for item in out:
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[1])
                self.assertEqual(item[1].get_swath_info().swath, swaths[1])

    def test_iter_channels_generator_filter_4(self) -> None:
        """Testing iter_channels_generator function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                content = content.replace(
                    "<Swath>XXXX</Swath>", f"<Swath>{swaths[1]}</Swath>"
                )
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")

            out = channel_iteration.iter_channels_generator(
                product=pf,
                filter_func=channel_iteration.SwathIDFilter(polarization=pols[:2]),
            )

            # check results
            self.assertIsInstance(out, GeneratorType)
            for idx, item in enumerate(out):
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[idx])

    def test_iter_channels_generator_filter_5(self) -> None:
        """Testing iter_channels_generator function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                content = content.replace(
                    "<Swath>XXXX</Swath>", f"<Swath>{swaths[1]}</Swath>"
                )
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")

            out = channel_iteration.iter_channels_generator(
                product=pf,
                filter_func=channel_iteration.SwathIDFilter(
                    polarization=pols[:2], swath=swaths[1]
                ),
            )

            # check results
            self.assertIsInstance(out, GeneratorType)
            for idx, item in enumerate(out):
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[idx])
                self.assertEqual(item[1].get_swath_info().swath, swaths[1])

    def test_iter_channels_filter_1(self) -> None:
        """Testing iter_channels function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")
            out = channel_iteration.iter_channels(product=pf, polarization=pols[2])

            # check results
            self.assertIsInstance(out, GeneratorType)
            for item in out:
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[2])

    def test_iter_channels_filter_2(self) -> None:
        """Testing iter_channels function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                content = content.replace(
                    "<Swath>XXXX</Swath>", f"<Swath>{swaths[2]}</Swath>"
                )
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")

            out = channel_iteration.iter_channels(product=pf, swath=swaths[2])

            # check results
            self.assertIsInstance(out, GeneratorType)
            for item in out:
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().swath, swaths[2])

    def test_iter_channels_filter_3(self) -> None:
        """Testing iter_channels function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                content = content.replace(
                    "<Swath>XXXX</Swath>", f"<Swath>{swaths[2]}</Swath>"
                )
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")

            out = channel_iteration.iter_channels(
                product=pf, polarization=pols[3], swath=swaths[2]
            )

            # check results
            self.assertIsInstance(out, GeneratorType)
            for item in out:
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[3])
                self.assertEqual(item[1].get_swath_info().swath, swaths[2])

    def test_iter_channels_filter_4(self) -> None:
        """Testing iter_channels function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                content = content.replace(
                    "<Swath>XXXX</Swath>", f"<Swath>{swaths[2]}</Swath>"
                )
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")

            out = channel_iteration.iter_channels(product=pf, polarization=pols[:3])

            # check results
            self.assertIsInstance(out, GeneratorType)
            for idx, item in enumerate(out):
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[idx])

            self.assertEqual(idx, 2)

    def test_iter_channels_filter_5(self) -> None:
        """Testing iter_channels function, with filter"""
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = create_product_folder(product_path)
            channel_paths = [
                product_path.joinpath(pf.pf_name + f"_000{c+1}.xml")
                for c in range(len(self.pol_xml))
            ]
            # writing metadata and raster files
            for item, content in zip(channel_paths, self.pol_xml):
                content = content.replace(
                    "<Swath>XXXX</Swath>", f"<Swath>{swaths[2]}</Swath>"
                )
                item.write_text(content)
                item.with_suffix("").write_bytes(b"")

            out = channel_iteration.iter_channels(
                product=pf, polarization=pols[:2], swath=swaths[2]
            )

            # check results
            self.assertIsInstance(out, GeneratorType)
            for idx, item in enumerate(out):
                self.assertIsInstance(item, tuple)
                self.assertIsInstance(item[0], int)
                self.assertIsInstance(item[1], MetaData)
                self.assertEqual(item[1].get_swath_info().polarization.value, pols[idx])
                self.assertEqual(item[1].get_swath_info().swath, swaths[2])

            self.assertEqual(idx, 1)


if __name__ == "__main__":
    unittest.main()
