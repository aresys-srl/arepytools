# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for io/product_folder_utils.py core functionalities"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import arepytools.io.productfolder2 as PF2
from arepytools.io.manifest import Manifest
from arepytools.io.productfolder_layout import (
    MANIFEST_NAME,
    METADATA_EXTENSION,
    ProductFolderLayout,
    QuicklookExtensions,
    RasterExtensions,
)


def _check_pf_validity(
    product: PF2.ProductFolder2,
    path: Path,
    extension: RasterExtensions = RasterExtensions.RAW,
) -> None:
    """Support function to check ProductFolder2 validity after creation.

    Parameters
    ----------
    product : PF2.ProductFolder2
        input ProductFolder2 object
    path : Path
        product folder path
    extensions : RasterExtensions
        raster extension to be checked, by default RasterExtensions.RAW
    """

    assert isinstance(product, PF2.ProductFolder2)
    assert isinstance(product.manifest, Path)
    assert isinstance(product.path, Path)
    assert isinstance(product.pf_name, str)
    assert isinstance(product._raster_extension, RasterExtensions)
    assert isinstance(product.raster_extension, str)
    assert product.path == path
    assert product.pf_name == path.name
    assert product.manifest == path.joinpath(MANIFEST_NAME)
    assert product.raster_extension == RasterExtensions(extension).value


def _check_pf_equivalence(pf1: PF2.ProductFolder2, pf2: PF2.ProductFolder2) -> None:
    """Checking the equivalence of two PFs.

    Parameters
    ----------
    pf1 : utils.ProductFolder2
        first ProductFolder2
    pf2 : utils.ProductFolder2
        second ProductFolder2
    """
    assert pf1.path == pf2.path
    assert pf1.raster_extension == pf2.raster_extension
    assert pf1.pf_name == pf2.pf_name
    assert pf1.manifest == pf2.manifest
    assert pf1.get_config_file() == pf2.get_config_file()
    assert pf1.get_overlay_file() == pf2.get_overlay_file()
    assert pf1.get_channel_metadata(3) == pf2.get_channel_metadata(3)
    assert pf1.get_channel_data(3) == pf2.get_channel_data(3)


class ProductFolderUtilitiesTest(unittest.TestCase):
    """Testing main Product Folder utilities"""

    def setUp(self):
        # two product folder names, one with a dot in it and one without
        self.product_names = ["TEST_SLC_01", "TEST.SLC_01", "TEST-SLC_01"]
        self.dummy_path = Path(r"C:\Users\user\data")
        self.r_extension = ".tiff"
        self.ql_extension = ".png"
        self.channels = [0, 4, 5, 8, 12, 35]
        self.manifest_name = "aresys_product"

    def test_pf_creation_private(self) -> None:
        """Testing ProductFolder2 object creation, default extension"""
        for name in self.product_names:
            path = self.dummy_path.joinpath(name)
            product_folder = PF2.ProductFolder2(path=path)

            # checking results
            self.assertIsInstance(product_folder._layout, ProductFolderLayout)
            # self.assertIsInstance(product_folder.raster_extension, str)
            self.assertIsInstance(product_folder._raster_extension, RasterExtensions)
            # self.assertIsInstance(product_folder.quicklook_extension, str)
            self.assertIsInstance(product_folder._path, Path)
            self.assertIsInstance(product_folder._manifest, Path)
            self.assertIsInstance(product_folder._pf_name, str)
            self.assertEqual(product_folder._path, path)
            self.assertEqual(product_folder._pf_name, name)
            self.assertEqual(product_folder._manifest, path.joinpath(MANIFEST_NAME))
            self.assertEqual(product_folder._raster_extension, RasterExtensions.RAW)

    def test_pf_creation_properties(self) -> None:
        """Testing ProductFolder2 object creation, default extension"""
        for name in self.product_names:
            path = self.dummy_path.joinpath(name)
            product_folder = PF2.ProductFolder2(path=path)

            # checking results
            _check_pf_validity(product=product_folder, path=path)

    def test_pf_creation_with_ext(self) -> None:
        """Testing ProductFolder2 object creation, with extension"""
        for name in self.product_names:
            path = self.dummy_path.joinpath(name)
            product_folder = PF2.ProductFolder2(
                path=path,
                raster_extension=self.r_extension,
            )

            # checking results
            _check_pf_validity(
                product=product_folder,
                path=path,
                extension=self.r_extension,
            )

    def test_pf_get_channels_list_error(self) -> None:
        """Testing ProductFolder2 get_channel_list method raising error"""
        with self.assertRaises(PF2.InvalidProductFolder):
            for name in self.product_names:
                path = self.dummy_path.joinpath(name)
                product_folder = PF2.ProductFolder2(path=path)
                product_folder.get_channels_list()

    def test_pf_get_channels_list_empty(self) -> None:
        """Testing ProductFolder2 get_channel_list, empty list"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                path.mkdir()
                product_folder = PF2.ProductFolder2(path=path)
                # dumping manifest file
                Manifest().write(product_folder.manifest)
                # reading channels
                channels = product_folder.get_channels_list()

                # checking results
                self.assertIsInstance(channels, list)
                self.assertEqual(len(channels), 0)

    def test_pf_get_channels_list(self) -> None:
        """Testing ProductFolder2 get_channel_list"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                path.mkdir()
                product_folder = PF2.ProductFolder2(path=path)
                # dumping manifest file
                Manifest().write(product_folder.manifest)
                # dumping metadata and rasters
                file_list = [product_folder.get_channel_data(c) for c in self.channels]
                metadata_list = [
                    product_folder.get_channel_metadata(c) for c in self.channels
                ]
                for ch_data in zip(file_list, metadata_list):
                    with open(ch_data[0], "w") as raster:
                        raster.write("")
                    with open(ch_data[1], "w") as meta:
                        meta.write("")
                channels = product_folder.get_channels_list()

                # checking results
                self.assertIsInstance(channels, list)
                self.assertEqual(len(channels), len(self.channels))
                self.assertEqual(sorted(channels), sorted(self.channels))

    def test_pf_get_channels_list_ext(self) -> None:
        """Testing ProductFolder2 get_channel_list, with extension"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                path.mkdir()
                product_folder = PF2.ProductFolder2(
                    path=path, raster_extension=self.r_extension
                )
                # dumping manifest file
                Manifest(datafile_extension=self.r_extension).write(
                    product_folder.manifest
                )
                # dumping metadata and rasters
                file_list = [product_folder.get_channel_data(c) for c in self.channels]
                metadata_list = [
                    product_folder.get_channel_metadata(c) for c in self.channels
                ]
                for ch_data in zip(file_list, metadata_list):
                    with open(ch_data[0], "w") as raster:
                        raster.write("")
                    with open(ch_data[1], "w") as meta:
                        meta.write("")
                channels = product_folder.get_channels_list()

                # checking results
                self.assertIsInstance(channels, list)
                self.assertEqual(len(channels), len(self.channels))
                self.assertEqual(sorted(channels), sorted(self.channels))


class CreateProductFolderTest(unittest.TestCase):
    """Testing create_product_folder function"""

    def setUp(self):
        # two product folder names, one with a dot in it and one without
        self.product_names = ["TEST_SLC_01", "TEST.SLC_01", "TEST-SLC_01"]
        self.r_extension = ".tiff"
        self.ql_extension = ".png"

    def test_create_product_folder_no_ext(self) -> None:
        """Testing creation of a product folder"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                product_folder = PF2.create_product_folder(pf_path=path)

                # checking results
                _check_pf_validity(product_folder, path)

    def test_create_product_folder_ext(self) -> None:
        """Testing creation of a product folder, with extension"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                product_folder = PF2.create_product_folder(
                    pf_path=path,
                    raster_extension=self.r_extension,
                )

                # checking results
                _check_pf_validity(
                    product_folder, path, extension=RasterExtensions(self.r_extension)
                )

    def test_create_product_folder_description(self) -> None:
        """Testing creation of a product folder, with description"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                description = "prova"
                path = Path(temp_dir).joinpath(name)
                product_folder = PF2.create_product_folder(
                    pf_path=path,
                    raster_extension=self.r_extension,
                    product_folder_description=description,
                )

                # checking manifest description
                manifest = Manifest.from_file(product_folder.manifest)
                self.assertEqual(manifest.description, description)

                # checking results
                _check_pf_validity(
                    product_folder, path, extension=RasterExtensions(self.r_extension)
                )

    def test_create_product_folder_overwrite(self) -> None:
        """Testing creation of a product folder, with overwrite"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                product_folder = PF2.create_product_folder(pf_path=path)

                # checking results
                _check_pf_validity(product_folder, path)

                product_folder_ovwr = PF2.create_product_folder(
                    pf_path=path, overwrite_ok=True
                )
                # checking results
                _check_pf_validity(
                    product_folder_ovwr,
                    path,
                )

    def test_create_product_folder_no_overwrite_error(self) -> None:
        """Testing creation of a product folder already existing but no overwrite"""
        with self.assertRaises(PF2.PathAlreadyExistsError):
            with TemporaryDirectory() as temp_dir:
                for name in self.product_names:
                    path = Path(temp_dir).joinpath(name)
                    path.mkdir()
                    PF2.create_product_folder(pf_path=path)

    def test_create_product_folder_overwrite_not_valid_error(self) -> None:
        """Testing creation of a product folder already existing with overwriting but invalid ProductFolder2"""
        with self.assertRaises(PF2.InvalidProductFolder):
            with TemporaryDirectory() as temp_dir:
                for name in self.product_names:
                    path = Path(temp_dir).joinpath(name)
                    product_folder = PF2.create_product_folder(pf_path=path)
                    product_folder.manifest.unlink()
                    PF2.create_product_folder(pf_path=path, overwrite_ok=True)


class OpenProductFolderTest(unittest.TestCase):
    """Testing open_product_folder function"""

    def setUp(self):
        # two product folder names, one with a dot in it and one without
        self.product_names = ["TEST_SLC_01", "TEST.SLC_01", "TEST-SLC_01"]
        self.r_extension = ".tiff"
        self.dummy_path = Path(r"C:\Users\user\data")

    def test_open_product_folder_no_ext(self) -> None:
        """Testing opening a product folder"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(pf_path=path)

                pf_opened = PF2.open_product_folder(pf_path=path)

                # checking results
                _check_pf_validity(pf_created, path)
                _check_pf_validity(pf_opened, path)
                _check_pf_equivalence(pf_created, pf_opened)

    def test_open_product_folder_ext(self) -> None:
        """Testing opening a product folder, with extension"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path,
                    raster_extension=self.r_extension,
                )

                pf_opened = PF2.open_product_folder(pf_path=path)

                # checking results
                _check_pf_validity(
                    pf_created, path, extension=RasterExtensions(self.r_extension)
                )
                _check_pf_validity(
                    pf_opened, path, extension=RasterExtensions(self.r_extension)
                )
                _check_pf_equivalence(pf_created, pf_opened)

    def test_open_product_folder_not_found_error(self) -> None:
        """Testing opening a product folder, not existent"""
        with self.assertRaises(PF2.ProductFolderNotFoundError):
            PF2.open_product_folder(pf_path=self.dummy_path)

    def test_open_product_folder_invalid_pf_error(self) -> None:
        """Testing opening a product folder, with extension"""
        with self.assertRaises(PF2.InvalidProductFolder):
            with TemporaryDirectory() as temp_dir:
                PF2.open_product_folder(pf_path=temp_dir)


class ProductFolder2UtilitiesTest(unittest.TestCase):
    """Testing ProductFolder2 utilities: is_valid, delete, rename"""

    def setUp(self):
        # two product folder names, one with a dot in it and one without
        self.product_names = ["TEST_SLC_01", "TEST.SLC_01", "TEST-SLC_01"]
        self.r_extension = ".tiff"
        self.ql_extension = QuicklookExtensions.JPG.value
        self.new_name = ["PROVA_SLC", "PRO.VA_SLC", "PRO-VA_SLC"]
        self.dummy_path = Path(r"C:\Users\user\data")
        self.channels = ["0001", "0009", "0034"]

    def test_rename_product_folder_no_ext_no_ql(self) -> None:
        """Testing rename_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for index, name in enumerate(self.product_names):
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path,
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel).write_text("", encoding="utf-8")
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )

                new_pf = path.with_name(self.new_name[index])
                self.assertFalse(new_pf.exists())

                PF2.rename_product_folder(current_folder=path, new_folder=new_pf)

                new = PF2.open_product_folder(new_pf)
                ch_list = new.get_channels_list()
                self.assertEqual(ch_list, [int(c) for c in self.channels])

    def test_rename_product_folder_ext_no_ql(self) -> None:
        """Testing rename_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for index, name in enumerate(self.product_names):
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path,
                    raster_extension=self.r_extension,
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel + self.r_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )

                new_pf = path.with_name(self.new_name[index])
                self.assertFalse(new_pf.exists())

                PF2.rename_product_folder(current_folder=path, new_folder=new_pf)

                new = PF2.open_product_folder(new_pf)
                ch_list = new.get_channels_list()
                self.assertEqual(ch_list, [int(c) for c in self.channels])

    def test_rename_product_folder_ext_ql(self) -> None:
        """Testing rename_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for index, name in enumerate(self.product_names):
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path,
                    raster_extension=self.r_extension,
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel + self.r_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + self.ql_extension).write_text(
                        "", encoding="utf-8"
                    )

                new_pf = path.with_name(self.new_name[index])
                self.assertFalse(new_pf.exists())

                PF2.rename_product_folder(current_folder=path, new_folder=new_pf)

                new = PF2.open_product_folder(new_pf)
                ch_list = new.get_channels_list()
                self.assertEqual(ch_list, [int(c) for c in self.channels])
                for channel in self.channels:
                    self.assertTrue(
                        new_pf.joinpath(
                            self.new_name[index] + "_" + channel + self.r_extension
                        ).exists()
                    )
                    self.assertTrue(
                        new_pf.joinpath(
                            self.new_name[index] + "_" + channel + self.ql_extension
                        ).exists()
                    )
                    self.assertTrue(
                        new_pf.joinpath(
                            self.new_name[index] + "_" + channel + METADATA_EXTENSION
                        ).exists()
                    )

    def test_rename_product_folder_other_files(self) -> None:
        """Testing rename_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for index, name in enumerate(self.product_names):
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path,
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel).write_text("", encoding="utf-8")
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )

                # adding other files
                path.joinpath("report.xml").write_text("", encoding="utf-8")
                path.joinpath("info.txt").write_text("", encoding="utf-8")
                path.joinpath("test.dat").write_text("", encoding="utf-8")

                new_pf = path.with_name(self.new_name[index])
                self.assertFalse(new_pf.exists())

                PF2.rename_product_folder(current_folder=path, new_folder=new_pf)

                new = PF2.open_product_folder(new_pf)
                ch_list = new.get_channels_list()
                self.assertEqual(ch_list, [int(c) for c in self.channels])
                self.assertTrue(new_pf.joinpath("report.xml").exists())
                self.assertTrue(new_pf.joinpath("info.txt").exists())
                self.assertTrue(new_pf.joinpath("test.dat").exists())

    def test_rename_product_folder_error1(self) -> None:
        """Testing rename_product_folder function, raising errors"""
        # error: existing new pf
        with self.assertRaises(PF2.InvalidProductFolder):
            with TemporaryDirectory() as temp_dir:
                for index, name in enumerate(self.product_names):
                    path = Path(temp_dir).joinpath(name)
                    pf_created = PF2.create_product_folder(
                        pf_path=path,
                        raster_extension=self.r_extension,
                    )
                    new_pf = path.with_name(self.new_name[index])
                    new_pf.mkdir()
                    PF2.rename_product_folder(current_folder=path, new_folder=new_pf)

    def test_rename_product_folder_error2(self) -> None:
        """Testing rename_product_folder function, raising errors"""
        # error: not existing old pf
        with self.assertRaises(PF2.InvalidProductFolder):
            with TemporaryDirectory() as temp_dir:
                for index, name in enumerate(self.product_names):
                    path = Path(temp_dir).joinpath(name)
                    new_pf = path.with_name(self.new_name[index])
                    PF2.rename_product_folder(current_folder=path, new_folder=new_pf)

    def test_rename_product_folder_error3(self) -> None:
        """Testing rename_product_folder function, raising errors"""
        # error: old pf is file
        with self.assertRaises(PF2.InvalidProductFolder):
            with TemporaryDirectory() as temp_dir:
                for index, name in enumerate(self.product_names):
                    path = Path(temp_dir).joinpath(name + ".xml")
                    path.write_text("", encoding="utf-8")
                    new_pf = path.with_name(self.new_name[index])
                    PF2.rename_product_folder(current_folder=path, new_folder=new_pf)

    def test_delete_product_folder_no_ext_no_ql(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path,
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel).write_text("", encoding="utf-8")
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )

                PF2.delete_product_folder_content(pf_created)

                # checking results
                self.assertFalse(path.exists())

    def test_delete_product_folder_ext_no_ql(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path, raster_extension=self.r_extension
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel + self.r_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )

                PF2.delete_product_folder_content(pf_created)

                # checking results
                self.assertFalse(path.exists())

    def test_delete_product_folder_ext_ql(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path, raster_extension=self.r_extension
                )
                # creating channel data and metadata files
                ql_files = []
                for channel in self.channels:
                    path.joinpath(name + "_" + channel + self.r_extension).write_text(
                        "", encoding="utf-8"
                    )
                    ql = path.joinpath(name + "_" + channel + self.ql_extension)
                    ql_files.append(ql)
                    ql.write_text("", encoding="utf-8")
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )

                PF2.delete_product_folder_content(pf_created)

                res_files = [f.name for f in pf_created.path.iterdir()]
                res_files_expected = [f.name for f in ql_files] + ["aresys_product"]

                # checking results
                self.assertTrue(path.exists())
                self.assertTrue(path.is_dir())
                self.assertEqual(len(res_files), len(res_files_expected))
                self.assertListEqual(sorted(res_files), sorted(res_files_expected))

    def test_delete_product_folder_other_files(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                pf_created = PF2.create_product_folder(
                    pf_path=path, raster_extension=self.r_extension
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel + self.r_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )
                path.joinpath("report.txt").write_text("", encoding="utf-8")

                PF2.delete_product_folder_content(pf_created)

                res_files = [f.name for f in pf_created.path.iterdir()]
                res_files_expected = ["aresys_product", "report.txt"]

                # checking results
                self.assertTrue(path.exists())
                self.assertTrue(path.is_dir())
                self.assertTrue(path.joinpath("report.txt").exists())
                self.assertEqual(len(res_files), len(res_files_expected))
                self.assertListEqual(sorted(res_files), sorted(res_files_expected))

    def test_is_valid_product_folder_no_files(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                PF2.create_product_folder(
                    pf_path=path,
                )
                self.assertTrue(PF2.is_valid_product_folder(path))

    def test_is_valid_product_folder_no_ql(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                PF2.create_product_folder(
                    pf_path=path,
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel).write_text("", encoding="utf-8")
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )
                self.assertTrue(PF2.is_valid_product_folder(path))

    def test_is_valid_product_folder(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                PF2.create_product_folder(
                    pf_path=path, raster_extension=self.r_extension
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel + self.r_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + self.ql_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )
                self.assertTrue(PF2.is_valid_product_folder(path))

    def test_is_valid_product_folder_other_file(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                PF2.create_product_folder(
                    pf_path=path, raster_extension=self.r_extension
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel + self.r_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + self.ql_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + METADATA_EXTENSION).write_text(
                        "", encoding="utf-8"
                    )
                path.joinpath("report.txt").write_text("", encoding="utf-8")
                self.assertTrue(PF2.is_valid_product_folder(path))

    def test_is_valid_product_folder_error(self) -> None:
        """Testing delete_product_folder function"""
        with TemporaryDirectory() as temp_dir:
            for name in self.product_names:
                path = Path(temp_dir).joinpath(name)
                PF2.create_product_folder(
                    pf_path=path, raster_extension=self.r_extension
                )
                # creating channel data and metadata files
                for channel in self.channels:
                    path.joinpath(name + "_" + channel + self.r_extension).write_text(
                        "", encoding="utf-8"
                    )
                    path.joinpath(name + "_" + channel + self.ql_extension).write_text(
                        "", encoding="utf-8"
                    )
                    if int(channel) % 2 == 1:
                        path.joinpath(
                            name + "_" + channel + METADATA_EXTENSION
                        ).write_text("", encoding="utf-8")

                self.assertFalse(PF2.is_valid_product_folder(path))


if __name__ == "__main__":
    unittest.main()
