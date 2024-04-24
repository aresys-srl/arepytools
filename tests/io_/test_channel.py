# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest
import warnings
from pathlib import Path

from arepytools.io import ChannelDeprecationWarning
from arepytools.io.channel import (
    Channel,
    _retrieve_extension_from_raster_file_name,
    _retrieve_metadata_file_name_from_raster_file_name,
)
from arepytools.io.metadata import MetaData

warnings.filterwarnings("ignore", category=ChannelDeprecationWarning)


class ChannelTestCase(unittest.TestCase):
    def test_not_existing_file(self):
        with self.assertRaises(FileNotFoundError):
            Channel("path/to/not/existing/raster_file_0001", MetaData(), "r")

    def test_wrong_metadata_type(self):
        with self.assertRaises(TypeError):
            Channel("file_name", "metadata", "w")

    def test_filenames(self):
        current_directory = Path().cwd()

        channel = Channel("name_only_0001", MetaData(), "w")

        self.assertEqual(channel.file_name, "name_only_0001")
        self.assertEqual(
            channel.raster_file, str(current_directory.joinpath("name_only_0001"))
        )
        self.assertEqual(
            channel.metadata_file, str(current_directory.joinpath("name_only_0001.xml"))
        )

        full_path = str(Path("full", "path_0001").absolute())
        channel = Channel(full_path, MetaData(), "w")

        self.assertEqual(channel.file_name, full_path)
        self.assertEqual(channel.raster_file, full_path)
        self.assertEqual(channel.metadata_file, full_path + ".xml")

        channel = Channel("relative/path_0001", MetaData(), "w")

        self.assertEqual(channel.file_name, "relative/path_0001")
        self.assertTrue(Path(channel.raster_file).is_absolute())
        self.assertEqual(
            channel.raster_file, str(current_directory.joinpath("relative/path_0001"))
        )
        self.assertTrue(Path(channel.metadata_file).is_absolute())
        self.assertEqual(
            channel.metadata_file,
            str(current_directory.joinpath("relative/path_0001.xml")),
        )

        channel = Channel("relative/path_0001.tiff", MetaData(), "w")

        self.assertEqual(channel.file_name, "relative/path_0001.tiff")
        self.assertTrue(Path(channel.raster_file).is_absolute())
        self.assertEqual(
            channel.raster_file,
            str(current_directory.joinpath("relative/path_0001.tiff")),
        )
        self.assertTrue(Path(channel.metadata_file).is_absolute())
        self.assertEqual(
            channel.metadata_file,
            str(current_directory.joinpath("relative/path_0001.xml")),
        )

        channel = Channel("relative/name_0001_path_0001.tiff", MetaData(), "w")

        self.assertEqual(channel.file_name, "relative/name_0001_path_0001.tiff")
        self.assertTrue(Path(channel.raster_file).is_absolute())
        self.assertEqual(
            channel.raster_file,
            str(current_directory.joinpath("relative/name_0001_path_0001.tiff")),
        )
        self.assertTrue(Path(channel.metadata_file).is_absolute())
        self.assertEqual(
            channel.metadata_file,
            str(current_directory.joinpath("relative/name_0001_path_0001.xml")),
        )

        channel = Channel("relative/name_0001_path_0001", MetaData(), "w")

        self.assertEqual(channel.file_name, "relative/name_0001_path_0001")
        self.assertTrue(Path(channel.raster_file).is_absolute())
        self.assertEqual(
            channel.raster_file,
            str(current_directory.joinpath("relative/name_0001_path_0001")),
        )
        self.assertTrue(Path(channel.metadata_file).is_absolute())
        self.assertEqual(
            channel.metadata_file,
            str(current_directory.joinpath("relative/name_0001_path_0001.xml")),
        )


class RetrieveExtensionTestCase(unittest.TestCase):
    def test_retrieve_extension(self):
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster_name_0001"), None
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster_name_1234"), None
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster.name_0001"), None
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster.file.name_0001"), None
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster_name_0001_0001"), None
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster_name_0001_test_0003"),
            None,
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("_0001_test_0003"), None
        )
        self.assertEqual(_retrieve_extension_from_raster_file_name("_0001"), None)
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster_name_0001.tiff"), ".tiff"
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster_name_1234.tiff"), ".tiff"
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster.name_0001.tiff"), ".tiff"
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster.file.name_0001.tiff"),
            ".tiff",
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("raster_name_0001_0001.tiff"),
            ".tiff",
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name(
                "raster_name_0001_test_0003.tiff"
            ),
            ".tiff",
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("_0001_test_0003.tiff"), ".tiff"
        )
        self.assertEqual(
            _retrieve_extension_from_raster_file_name("_0001.tiff"), ".tiff"
        )

    def test_raises_on_invalid_name(self):
        self.assertRaises(
            RuntimeError,
            _retrieve_extension_from_raster_file_name,
            "raster.file.name",
        )
        self.assertRaises(
            RuntimeError,
            _retrieve_extension_from_raster_file_name,
            "raster.file.name_01",
        )
        self.assertRaises(
            RuntimeError,
            _retrieve_extension_from_raster_file_name,
            "raster.file.name.tiff",
        )
        self.assertRaises(
            RuntimeError,
            _retrieve_extension_from_raster_file_name,
            "raster.file.name_01.tiff",
        )


class RetrieveMetadataFileNameTestCase(unittest.TestCase):
    def test_retrieve_extension(self):
        self.assertEqual(
            _retrieve_metadata_file_name_from_raster_file_name("raster_name_0001"),
            "raster_name_0001.xml",
        )
        self.assertEqual(
            _retrieve_metadata_file_name_from_raster_file_name("raster_name_1234"),
            "raster_name_1234.xml",
        )
        self.assertEqual(
            _retrieve_metadata_file_name_from_raster_file_name("raster.name_0001"),
            "raster.name_0001.xml",
        )
        self.assertEqual(
            _retrieve_metadata_file_name_from_raster_file_name("raster.file.name_0001"),
            "raster.file.name_0001.xml",
        )
        self.assertEqual(
            _retrieve_metadata_file_name_from_raster_file_name("raster_name_0001.tiff"),
            "raster_name_0001.xml",
        )
        self.assertEqual(
            _retrieve_metadata_file_name_from_raster_file_name("raster_name_1234.tiff"),
            "raster_name_1234.xml",
        )
        self.assertEqual(
            _retrieve_metadata_file_name_from_raster_file_name("raster.name_0001.tiff"),
            "raster.name_0001.xml",
        )
        self.assertEqual(
            _retrieve_metadata_file_name_from_raster_file_name(
                "raster.file.name_0001.tiff"
            ),
            "raster.file.name_0001.xml",
        )


if __name__ == "__main__":
    unittest.main()
