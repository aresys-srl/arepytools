# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for io/product_folder_utils.py core functionalities"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from arepytools.io.manifest import Manifest, RasterExtensions
from arepytools.io.productfolder_layout import MANIFEST_NAME


class ManifestTest(unittest.TestCase):
    """Testing Manifest class"""

    def setUp(self):
        self.extension = ".tiff"
        self.version = "2.1"
        self.description = "ProductFolder initialized by ArePyTools"

    def test_manifest_no_ext(self) -> None:
        """Testing Manifest dataclass"""
        manifest = Manifest()
        self.assertEqual(manifest.version, self.version)
        self.assertIsNone(manifest.description)
        self.assertIsInstance(manifest.datafile_extension, RasterExtensions)
        self.assertEqual(manifest.datafile_extension, RasterExtensions.RAW)

    def test_manifest_no_ext_with_description(self) -> None:
        """Testing Manifest dataclass"""
        manifest = Manifest(description=self.description)
        self.assertEqual(manifest.version, self.version)
        self.assertEqual(manifest.description, self.description)
        self.assertIsInstance(manifest.datafile_extension, RasterExtensions)
        self.assertEqual(manifest.datafile_extension, RasterExtensions.RAW)

    def test_manifest_ext(self) -> None:
        """Testing Manifest dataclass, with extension"""
        manifest = Manifest(datafile_extension=self.extension)
        self.assertEqual(manifest.version, self.version)
        self.assertIsNone(manifest.description)
        self.assertIsInstance(manifest.datafile_extension, RasterExtensions)
        self.assertTrue(manifest.datafile_extension == RasterExtensions(self.extension))

    def test_manifest_write(self) -> None:
        """Testing Manifest dump to disk"""
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir).joinpath(MANIFEST_NAME)
            assert not manifest_path.exists()
            Manifest().write(manifest_path)
            assert manifest_path.exists()
            assert manifest_path.is_file()

    def test_manifest_read(self) -> None:
        """Testing Manifest dump to disk"""
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir).joinpath(MANIFEST_NAME)
            Manifest(description=self.description).write(manifest_path)
            manifest = Manifest.from_file(manifest_path)
            self.assertIsInstance(manifest, Manifest)
            self.assertEqual(manifest.version, self.version)
            self.assertEqual(manifest.description, self.description)
            self.assertIsInstance(manifest.datafile_extension, RasterExtensions)
            self.assertEqual(manifest.datafile_extension, RasterExtensions.RAW)

    def test_manifest_read_ext(self) -> None:
        """Testing Manifest dump to disk, with extension"""
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir).joinpath(MANIFEST_NAME)
            Manifest(datafile_extension=self.extension).write(manifest_path)
            manifest = Manifest.from_file(manifest_path)
            self.assertIsInstance(manifest, Manifest)
            self.assertEqual(manifest.version, self.version)
            self.assertIsNone(manifest.description)
            self.assertIsInstance(manifest.datafile_extension, RasterExtensions)
            self.assertEqual(
                manifest.datafile_extension, RasterExtensions(self.extension)
            )


if __name__ == "__main__":
    unittest.main()
