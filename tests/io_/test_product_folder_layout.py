# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for io/product_folder_layout.py core functionalities"""

import unittest
from pathlib import Path

from arepytools.io.productfolder_layout import (
    InvalidChannelNumber,
    ProductFolderLayout,
    QuicklookExtensions,
    RasterExtensions,
)


class ProductFolderLayoutTest(unittest.TestCase):
    """Testing ProductFolderLayout class"""

    def setUp(self):
        # two product folder names, one with a dot in it and one without
        self.product_names = ["TEST_SLC_01", "TEST.SLC_01"]
        base_folder = r"C:\Users\user\data"
        self.path = [Path(base_folder, name) for name in self.product_names]
        self.default_manifest_name = "aresys_product"
        self.channel = 3

        self.results = [
            {
                "manifest": Path(base_folder, name, self.default_manifest_name),
                "config": Path(base_folder, name, name + ".config"),
                "kmz": Path(base_folder, name, name + ".kmz"),
                "channel_data_raw": Path(
                    base_folder, name, name + f"_{self.channel:04d}"
                ),
                "channel_data_tiff": Path(
                    base_folder, name, name + f"_{self.channel:04d}" + ".tiff"
                ),
                "channel_metadata": Path(
                    base_folder, name, name + f"_{self.channel:04d}" + ".xml"
                ),
                "channel_quicklook_png": Path(
                    base_folder, name, name + f"_{self.channel:04d}" + ".png"
                ),
                "channel_quicklook_jpg": Path(
                    base_folder, name, name + f"_{self.channel:04d}" + ".jpg"
                ),
            }
            for name in self.product_names
        ]

    def test_product_folder_layout_init_defaults(self) -> None:
        """Testing object initialization with defaults"""

        # instantiating object with extension
        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            # checking init variables
            self.assertIsInstance(layout, ProductFolderLayout)
            self.assertEqual(layout._pf_path, self.path[index])
            self.assertEqual(layout._product_name, self.path[index].name)

    def test_product_folder_layout_init_enum(self) -> None:
        """Testing object initialization with enum extensions"""

        # instantiating object with extension
        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            # checking init variables
            self.assertIsInstance(layout, ProductFolderLayout)
            self.assertEqual(layout._pf_path, self.path[index])
            self.assertEqual(layout._product_name, self.path[index].name)

    def test_product_folder_layout_generate_manifest(self) -> None:
        """Testing generate manifest method"""

        # testing product folder name with both a dot in it and without
        for index in range(2):
            self.assertEqual(
                ProductFolderLayout.generate_manifest_path(self.path[index]),
                self.results[index]["manifest"],
            )

    def test_product_folder_layout_get_config(self) -> None:
        """Testing get config method"""

        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            self.assertEqual(layout.get_config_path(), self.results[index]["config"])

    def test_product_folder_layout_get_kmz(self) -> None:
        """Testing get kmz method"""

        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            self.assertEqual(
                layout.get_overlay_path(),
                self.results[index]["kmz"],
            )

    def test_product_folder_layout_get_channel_metadata(self) -> None:
        """Testing get channel metadata method"""

        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            self.assertEqual(
                layout.get_channel_metadata_path(self.channel),
                self.results[index]["channel_metadata"],
            )

    def test_product_folder_layout_get_channel_data_raw(self) -> None:
        """Testing get channel data method"""

        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            self.assertEqual(
                layout.get_channel_data_path(
                    self.channel, extension=RasterExtensions.RAW
                ),
                self.results[index]["channel_data_raw"],
            )

    def test_product_folder_layout_get_channel_data_tiff(self) -> None:
        """Testing get channel data method"""

        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            self.assertEqual(
                layout.get_channel_data_path(
                    self.channel, extension=RasterExtensions.TIFF
                ),
                self.results[index]["channel_data_tiff"],
            )

    def test_product_folder_layout_get_channel_quicklook_png(self) -> None:
        """Testing get channel quicklook method, png format"""

        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            self.assertEqual(
                layout.get_channel_quicklook_path(
                    self.channel, extension=QuicklookExtensions.PNG
                ),
                self.results[index]["channel_quicklook_png"],
            )

    def test_product_folder_layout_get_channel_quicklook_jpg(self) -> None:
        """Testing get channel quicklook method, jpg format"""

        # testing product folder name with both a dot in it and without
        for index in range(2):
            layout = ProductFolderLayout(self.path[index])
            self.assertEqual(
                layout.get_channel_quicklook_path(
                    self.channel, extension=QuicklookExtensions.JPG
                ),
                self.results[index]["channel_quicklook_jpg"],
            )

    def test_raising_channel_number_error_negative(self) -> None:
        """Testing error raising for negative channel numbers"""

        with self.assertRaises(InvalidChannelNumber):
            for index in range(2):
                layout = ProductFolderLayout(self.path[index])
                layout.get_channel_quicklook_path(
                    -56, extension=QuicklookExtensions.JPG
                )

    def test_raising_channel_number_error_above_limit(self) -> None:
        """Testing error raising for channel number above max limit"""

        with self.assertRaises(InvalidChannelNumber):
            for index in range(2):
                layout = ProductFolderLayout(self.path[index])
                layout.get_channel_quicklook_path(
                    10000, extension=QuicklookExtensions.JPG
                )


if __name__ == "__main__":
    unittest.main()
