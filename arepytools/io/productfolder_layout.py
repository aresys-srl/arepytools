# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Product Folder Layout module
----------------------------
"""

from enum import Enum
from pathlib import Path
from typing import Union


class RasterExtensions(Enum):
    """Supported product folder raster extensions"""

    TIFF = ".tiff"
    RAW = ""


class QuicklookExtensions(Enum):
    """Supported product folder quicklook extensions"""

    PNG = ".png"
    JPG = ".jpg"


METADATA_EXTENSION = ".xml"
CONFIG_EXTENSION = ".config"
OVERLAY_EXTENSION = ".kmz"
_MAX_CHANNEL_NUMBER = 9999
MANIFEST_NAME = "aresys_product"


class InvalidChannelNumber(ValueError):
    """Channel number id provided exceeds validity boundaries"""


class ProductFolderLayout:
    """Class that manages the whole product folder layout and file paths"""

    def __init__(self, path: Union[str, Path]) -> None:
        """Product folder layout definition.
        This class can build full paths to channel dependent raster files (with specified extension), metadata files
        (with default extension), quicklook files (with specified extension) and product folder related config file and
        overlay (.kmz) file.

        Parameters
        ----------
        path : Union[str, Path]
            path to the product folder
        """
        self._pf_path = Path(path)
        self._product_name = self._pf_path.name

    @property
    def product_name(self) -> str:
        """Accessing Product Folder base name"""
        return self._product_name

    @staticmethod
    def generate_manifest_path(pf_path: Path) -> Path:
        """Generating the manifest file full path for the current Product Folder.

        Parameters
        ----------
        pf_path : Path
            Product Folder absolute base path

        Returns
        -------
        Path
            full path to manifest file
        """
        return pf_path.joinpath(MANIFEST_NAME)

    def _format_channel(self, channel_id: int) -> str:
        """Formatting product folder name to append the provided channel id.

        Parameters
        ----------
        channel_id : int
            channel number

        Returns
        -------
        str
            formatted channel name

        Raises
        ------
        InvalidChannelNumber
            if channel id is not an integer value
        InvalidChannelNumber
            if channel id is above maximum channel number
        InvalidChannelNumber
            if channel id is negative
        """
        try:
            channel_id = int(channel_id)
        except ValueError as exc:
            raise InvalidChannelNumber(
                f"Channel id {channel_id} is not an integer"
            ) from exc

        if channel_id > _MAX_CHANNEL_NUMBER:
            raise InvalidChannelNumber(
                f"Channel id {channel_id} is above upper boundary {_MAX_CHANNEL_NUMBER}"
            )

        if channel_id < 0:
            raise InvalidChannelNumber("Negative channel numbers are not supported")

        return self._product_name + f"_{channel_id:04d}"

    def get_config_path(self) -> Path:
        """Retrieving config full path.

        Returns
        -------
        Path
            path to the product folder's config file
        """
        config = self._product_name + CONFIG_EXTENSION
        return self._pf_path.joinpath(config)

    def get_overlay_path(self) -> Path:
        """Retrieving the overlay kmz archive full path.

        Returns
        -------
        Path
            path to the product folder's overlay file (.kmz archive)
        """
        kmz_archive = self._product_name + OVERLAY_EXTENSION
        return self._pf_path.joinpath(kmz_archive)

    def get_channel_metadata_path(self, channel_id: int) -> Path:
        """Retrieving the channel metadata full path for the given channel number.

        Parameters
        ----------
        channel_id : int
            channel number of choice

        Returns
        -------
        Path
            path to the channel's metadata file for the channel of choice
        """
        channel_metadata = self._format_channel(channel_id) + METADATA_EXTENSION
        return self._pf_path.joinpath(channel_metadata)

    def get_channel_data_path(
        self, channel_id: int, extension: RasterExtensions
    ) -> Path:
        """Retrieving the channel data full path for the given channel number.

        Parameters
        ----------
        channel_id : int
            channel number of choice
        extension : RasterExtensions
            extension of raster data file in folder

        Returns
        -------
        Path
            full path to the channel's data file for the channel of choice
        """
        channel_data = self._format_channel(channel_id) + extension.value
        return self._pf_path.joinpath(channel_data)

    def get_channel_quicklook_path(
        self, channel_id: int, extension: QuicklookExtensions
    ) -> Path:
        """Retrieving the channel quicklook full path for the given channel number.

        Parameters
        ----------
        channel_id : int
            channel number of choice
        extension : QuicklookExtensions
            extension of quicklook image file in folder

        Returns
        -------
        Path
            full path to the channel's quicklook file for the channel of choice
        """
        quick_look = self._format_channel(channel_id) + extension.value
        return self._pf_path.joinpath(quick_look)
