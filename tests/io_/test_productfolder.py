# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import os
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from arepytools.io import ProductFolderDeprecationWarning
from arepytools.io.channel import EOpenMode
from arepytools.io.metadata import EPolarization, SwathInfo
from arepytools.io.productfolder import IsNotAProductFolder
from arepytools.io.productfolder import ProductFolder as PF
from arepytools.io.productfolder import (
    ReadOnlyProductFolder,
    get_channel_indexes,
    remove_product_folder,
    rename_product_folder,
)
from arepytools.timing.precisedatetime import PreciseDateTime

warnings.filterwarnings("ignore", category=ProductFolderDeprecationWarning)


class ProductFolderTest(unittest.TestCase):
    def test_openmode_create_or_overwrite(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")

            self.assertEqual(pf.open_mode, EOpenMode.create_or_overwrite)
            with self.assertRaises(RuntimeError):
                pf = PF(product_path, "w")

    def test_openmode_create(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "x")

            self.assertEqual(pf.open_mode, EOpenMode.create)
            with self.assertRaises(RuntimeError):
                pf = PF(product_path, "x")

    def test_openmode_open(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")  # create
            pf = PF(product_path, "r")  # open
            self.assertEqual(pf.open_mode, EOpenMode.open)

    def test_openmode_open_product_with_content(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")  # create
            for ch_index in range(2):
                pf.append_channel(1, 1, "FLOAT64")
                pf.write_data(ch_index, np.array([[0]]))
                pf.write_metadata(ch_index)
            with open(pf.config_file, "w") as config_file:
                config_file.write("config_file_example")

            pf = PF(product_path, "r")  # open
            self.assertEqual(pf.open_mode, EOpenMode.open)

    def test_open_product_with_string(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")  # create
            for ch_index in range(2):
                pf.append_channel(1, 1, "FLOAT64")
                pf.write_data(ch_index, np.array([[0]]))
                pf.write_metadata(ch_index)
            with open(pf.config_file, "w") as config_file:
                config_file.write("config_file_example")

            str_path = str(product_path) + os.sep
            pf = PF(str_path, "r")  # open
            self.assertEqual(pf.get_number_channels(), 2)

    def test_openmode_open_failure(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")  # create
            Path(pf._manifest_file).unlink()
            with self.assertRaises(IsNotAProductFolder):
                PF(product_path, "r")  # open

    def test_openmode_open_prevent_appending_channel(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")  # create
            pf = PF(product_path, "r")  # open
            with self.assertRaises(ReadOnlyProductFolder):
                pf.append_channel(1, 1, "FLOAT64")

    def test_openmode_open_prevent_overwriting(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")  # create
            pf.append_channel(1, 1, "FLOAT64")
            pf.write_data(0, np.array([[0]]))
            pf.write_metadata(0)

            pf = PF(product_path, "r")  # open
            with self.assertRaises(ReadOnlyProductFolder):
                pf.write_data(0, np.array([[1]]))

            with self.assertRaises(ReadOnlyProductFolder):
                pf.write_metadata(0)

    def test_openmode_open_or_create(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            with self.assertRaises(NotImplementedError):
                pf = PF(product_path, "a")  # open, it does not exist, it creates it
                pf = PF(product_path, "a")  # open it
                self.assertEqual(pf.open_mode, EOpenMode.open_or_create)

    def test_properties(self):
        with TemporaryDirectory() as tmpdir:
            names = [
                "test_product",
                "test.product",
                "test-product",
                "test.product.extension",
            ]
            for name in names:
                product_path = Path(tmpdir, name)
                pf = PF(product_path, "w")

                pf.append_channel(1, 1, "FLOAT64")
                pf.write_data(0, np.array([[0]]))
                pf.write_metadata(0)

                config_file = str(product_path.joinpath(name + ".config").absolute())
                manifest_file = product_path.joinpath("aresys_product").absolute()

                self.assertTrue(PF.is_productfolder(product_path))
                self.assertEqual(pf.pf_name, name)
                self.assertEqual(pf.pf_dir_path, str(product_path))
                self.assertEqual(pf.config_file, config_file)
                self.assertEqual(pf._manifest_file, manifest_file)

    def test_properties_tiff(self):
        with TemporaryDirectory() as tmpdir:
            names = [
                "test_product",
                "test.product",
                "test-product",
                "test.product.extension",
            ]
            for name in names:
                product_path = Path(tmpdir, name)
                product_path.mkdir()

                product_path.joinpath(name + "_0001.tiff").touch()
                product_path.joinpath(name + "_0001.xml").touch()
                manifest_file = product_path.joinpath("aresys_product").absolute()
                PF._create_manifest(manifest_file, ".tiff")

                self.assertTrue(PF.is_productfolder(product_path))

    def test_empty_product_valid(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            PF(product_path, "w")
            self.assertTrue(PF.is_productfolder(product_path))

    def test_one_channel_product_valid(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")
            pf.append_channel(1, 1, "FLOAT64")
            pf.write_data(0, np.array([[0]]))
            pf.write_metadata(0)

            self.assertTrue(PF.is_productfolder(product_path))

    def test_two_channels_product_valid(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")
            for ch_index in range(2):
                pf.append_channel(1, 1, "FLOAT64")
                pf.write_data(ch_index, np.array([[0]]))
                pf.write_metadata(ch_index)

            self.assertTrue(PF.is_productfolder(product_path))

    def test_missing_metadata_invalid(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")
            for ch_index in range(2):
                pf.append_channel(1, 1, "FLOAT64")
                pf.write_data(ch_index, np.array([[0]]))
                if ch_index != 1:
                    pf.write_metadata(ch_index)

            self.assertFalse(PF.is_productfolder(product_path))

    def test_missing_data_invalid(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")
            for ch_index in range(3):
                pf.append_channel(1, 1, "FLOAT64")
                if ch_index != 0:
                    pf.write_data(ch_index, np.array([[0]]))
                pf.write_metadata(ch_index)

            self.assertFalse(PF.is_productfolder(product_path))

    def test_missing_manifest_invalid(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            pf = PF(product_path, "w")
            for ch_index in range(3):
                pf.append_channel(1, 1, "FLOAT64")
                pf.write_data(ch_index, np.array([[0]]))
                pf.write_metadata(ch_index)
            Path(pf._manifest_file).unlink()

            self.assertFalse(PF.is_productfolder(product_path))

    def test_file_is_invalid(self):
        with TemporaryDirectory() as tmpdir:
            product_path = Path(tmpdir, "test_product")
            with open(product_path, "w") as test_file:
                test_file.write("file_product")

            self.assertFalse(PF.is_productfolder(product_path))


class HelperFunctionsTest(unittest.TestCase):
    _test_product_name = "test_product"
    _swath_ids = [
        ("beam1", "H/H"),
        ("beam1", "H/V"),
        ("beam1", "H/H"),
        ("beam1", "H/H"),
        ("beam2", "V/V"),
        ("beam2", "V/V"),
        ("beam2", "H/H"),
        ("beam2", "V/V"),
        ("beam2", "H/H"),
    ]

    @classmethod
    def create_test_product(cls, directory, product_name, swath_ids) -> str:
        product_path = Path(directory, product_name)
        pf = PF(product_path, "w")

        for ch_index, (swath, pol) in enumerate(swath_ids):
            pf.append_channel(1, 1, "FLOAT64")
            pf.write_data(ch_index, np.array([[0]]))

            swath_info = SwathInfo(swath_i=swath, polarization_i=pol)
            swath_info.acquisition_start_time = PreciseDateTime()

            pf.get_channel(ch_index).metadata.get_metadata_channels(0).insert_element(
                swath_info
            )
            pf.write_metadata(ch_index)

        return str(product_path)

    def test_filter_on_swath_and_on_polarization(self):
        with TemporaryDirectory() as tmpdir:
            product_path = HelperFunctionsTest.create_test_product(
                tmpdir, self._test_product_name, self._swath_ids
            )
            pf = PF(product_path, "r")

            self.assertEqual(
                get_channel_indexes(pf, "beam1", EPolarization.hh),
                [0, 2, 3],
            )
            self.assertEqual(
                get_channel_indexes(pf, "beam1", "H/H"),
                [0, 2, 3],
            )

            self.assertEqual(
                get_channel_indexes(pf, "beam1", EPolarization.hv),
                [1],
            )
            self.assertEqual(
                get_channel_indexes(pf, swath_name="beam1", polarization="H/V"), [1]
            )

            self.assertEqual(
                get_channel_indexes(pf, "beam4", EPolarization.hh),
                [],
            )
            self.assertEqual(
                get_channel_indexes(pf, swath_name="beam4", polarization="H/H"),
                [],
            )

            self.assertEqual(
                get_channel_indexes(pf, "beam1", EPolarization.vv),
                [],
            )
            self.assertEqual(
                get_channel_indexes(pf, swath_name="beam1", polarization="V/V"),
                [],
            )

    def test_filter_on_swath(self):
        with TemporaryDirectory() as tmpdir:
            product_path = HelperFunctionsTest.create_test_product(
                tmpdir, self._test_product_name, self._swath_ids
            )
            pf = PF(product_path, "r")

            self.assertEqual(
                get_channel_indexes(pf, swath_name="beam2"), [4, 5, 6, 7, 8]
            )
            self.assertEqual(get_channel_indexes(pf, "beam1"), [0, 1, 2, 3])

            self.assertEqual(
                get_channel_indexes(pf, swath_name="beam1", polarization=None),
                [0, 1, 2, 3],
            )

            self.assertEqual(get_channel_indexes(pf, swath_name="beam4"), [])

    def test_filter_on_polarization(self):
        with TemporaryDirectory() as tmpdir:
            product_path = HelperFunctionsTest.create_test_product(
                tmpdir, self._test_product_name, self._swath_ids
            )
            pf = PF(product_path, "r")

            self.assertEqual(
                get_channel_indexes(pf, polarization=EPolarization.hh), [0, 2, 3, 6, 8]
            )
            self.assertEqual(
                get_channel_indexes(pf, polarization="H/H"), [0, 2, 3, 6, 8]
            )

            self.assertEqual(
                get_channel_indexes(pf, polarization=EPolarization.hv), [1]
            )
            self.assertEqual(get_channel_indexes(pf, polarization="H/V"), [1])

            self.assertEqual(
                get_channel_indexes(pf, polarization=EPolarization.vv), [4, 5, 7]
            )
            self.assertEqual(get_channel_indexes(pf, polarization="V/V"), [4, 5, 7])

            self.assertEqual(get_channel_indexes(pf, polarization=EPolarization.vh), [])
            self.assertEqual(get_channel_indexes(pf, polarization="V/H"), [])

            with self.assertRaises(ValueError):
                get_channel_indexes(pf, polarization="HH")

    def test_rename_product(self):
        with TemporaryDirectory() as tmpdir:
            in_product = HelperFunctionsTest.create_test_product(
                tmpdir, self._test_product_name, self._swath_ids
            )
            pf = PF(in_product, "r")

            with open(Path(pf.config_file), "w") as config_file:
                config_file.write("A config file")

            new_product = rename_product_folder(pf.pf_dir_path, "renamed_product")

            self.assertFalse(Path(in_product).exists())

            pf = PF(new_product, "r")
            self.assertEqual(pf.pf_name, "renamed_product")
            for ch_index in range(pf.get_number_channels()):
                raster_info = pf.get_channel(ch_index).get_raster_info(0)
                self.assertTrue("renamed_product" in raster_info.file_name)

            self.assertTrue(Path(pf.config_file).exists())

    def test_remove_product(self):
        with TemporaryDirectory() as tmpdir:
            in_product = HelperFunctionsTest.create_test_product(
                tmpdir, self._test_product_name, self._swath_ids
            )
            remove_product_folder(in_product)
            self.assertFalse(Path(in_product).exists())

    def test_remove_product_leave_extra_files(self):
        with TemporaryDirectory() as tmpdir:
            in_product = HelperFunctionsTest.create_test_product(
                tmpdir, self._test_product_name, self._swath_ids
            )

            extra_file_path = Path(in_product).joinpath("extra_file")
            with open(extra_file_path, "w") as extra_file:
                extra_file.write("Extra file")

            remove_product_folder(in_product)

            self.assertTrue(extra_file_path.exists())


if __name__ == "__main__":
    unittest.main()
