# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT


"""
**Deprecated** ProductFolder module
-----------------------------------

.. deprecated:: v1.6.0
    Using productfolder module from arepytools.io is deprecated, use productfolder2 instead
"""


import copy
import re
import warnings
from pathlib import Path
from typing import List, Optional, Union

from arepytools.io import (
    ProductFolderDeprecationWarning,
    RemoveProductFolderDeprecationWarning,
    RenameProductFolderDeprecationWarning,
    channel,
    read_metadata,
)

from .manifest import Manifest
from .metadata import EPolarization, RasterInfo

_header_extension = ".xml"


warnings.warn(
    "productfolder module is deprecated starting from Arepytools v1.6.0"
    + " please use productfolder2 instead",
    ProductFolderDeprecationWarning,
    stacklevel=2,
)


class ProductFolder:
    """
    ProductFolder class
    """

    def __init__(
        self,
        dir_path: Union[str, Path],
        open_mode: channel.EOpenModeLike = channel.EOpenMode.open,
    ):
        self._open_mode = channel.EOpenMode(open_mode)
        self.__pf_dir_path = Path(dir_path).absolute()
        self.__pf_name = Path(dir_path).name
        self.__channels = list()

        if self._open_mode == channel.EOpenMode.open:
            if not self.is_productfolder(str(self.__pf_dir_path)):
                raise IsNotAProductFolder(self.__pf_dir_path)
            self.__pf_read(self.__pf_dir_path, self._open_mode)

        elif (
            self._open_mode == channel.EOpenMode.create
            or self._open_mode == channel.EOpenMode.create_or_overwrite
        ):
            if self.__pf_dir_path.exists():
                raise RuntimeError(
                    "Cannot initialize {} on {}: path already exists".format(
                        self.__class__.__name__, self.__pf_dir_path
                    )
                )

            # Initialize the product folder
            self.__pf_generate(self.__pf_dir_path)
        else:
            raise NotImplementedError("{} mode not supported".format(self._open_mode))

        warnings.warn(
            "ProductFolder class is deprecated starting from Arepytools v1.6.0"
            + "please use productfolder2.open_product_folder or productfolder2.create_product_folder instead",
            ProductFolderDeprecationWarning,
            stacklevel=2,
        )

    @property
    def open_mode(self):
        return self._open_mode

    @property
    def pf_name(self) -> str:
        return self.__pf_name

    @property
    def pf_dir_path(self) -> str:
        return str(self.__pf_dir_path)

    @property
    def _manifest_file(self) -> Path:
        return self._get_manifest_file_path(self.__pf_dir_path)

    @property
    def config_file(self) -> str:
        return str(self.__pf_dir_path.joinpath(self.pf_name + ".config"))

    def __pf_generate(self, dir_path: Path):
        # Create the directory
        dir_path.mkdir()
        # Create the manifest
        self._create_manifest(self._manifest_file)

    @staticmethod
    def _get_manifest_file_path(product_dir: Path) -> Path:
        return product_dir.joinpath("aresys_product")

    @staticmethod
    def _create_manifest(
        manifest_file: Path, data_file_extension: Optional[str] = None
    ):
        manifest = Manifest(datafile_extension=data_file_extension)
        manifest.write(manifest_file)

    def __pf_read(self, dir_path: Path, open_mode: channel.EOpenMode):
        headers = self._get_header_files_in_dir(dir_path)
        raster_extension = self.get_data_file_extension(str(dir_path))
        for header in headers:
            metadata = read_metadata(dir_path.joinpath(header))
            raster_filename = header[0 : -len(_header_extension)] + raster_extension
            raster_path = dir_path.joinpath(raster_filename)
            self.__channels.append(
                channel.Channel(str(raster_path), metadata, open_mode)
            )

    def append_channel(
        self,
        lines,
        samples,
        data_type,
        header_offset=0,
        row_prefix=0,
        byte_order="LITTLEENDIAN",
    ):
        """
        Append a channel to the ProductFolder

        :param lines: number of lines
        :param samples: number of samples
        :param data_type: raster data type (see :class:`arepytools.io.metadata.ECellType`)
        :param header_offset: header offset in the raster
        :param row_prefix: row prefix in the raster
        :param byte_order: byte order of the raster (see :class:`arepytools.io.metadata.EByteOrder`)
        """
        if self._open_mode == channel.EOpenMode.open:
            raise ReadOnlyProductFolder
        # Define the channel file name
        current_num_of_channels = self.get_number_channels()
        pf_name = self.__pf_name
        suffix = "_{:0>4}".format(current_num_of_channels + 1)
        chan_file_path = self.__pf_dir_path.joinpath(pf_name + suffix)

        # Create the metadata
        file_name = chan_file_path.name
        current_raster_info = RasterInfo(
            lines, samples, data_type, file_name, header_offset, row_prefix, byte_order
        )

        # Create the new channel
        current_channel = channel.Channel.from_raster_info(
            str(chan_file_path), current_raster_info, self._open_mode
        )

        # Add the channel to the product folder
        self.__channels.append(current_channel)

    @staticmethod
    def _get_header_files_in_dir(dir_path: Path) -> list:
        pf_name = dir_path.name
        header_list = [
            f.name
            for f in dir_path.iterdir()
            if re.fullmatch(
                "{}".format(pf_name)
                + channel.CHANNEL_ID_RE_PATTERN
                + _header_extension,
                f.name,
            )
            is not None
        ]
        header_list.sort()
        return header_list

    @staticmethod
    def _get_raster_files_in_dir(dir_path: Path) -> list:
        pf_name = dir_path.name
        raster_list = [
            f.name
            for f in dir_path.iterdir()
            if re.fullmatch(
                "{}".format(pf_name) + channel.CHANNEL_ID_RE_PATTERN, f.name
            )
            is not None
        ]
        raster_list.sort()
        return raster_list

    @classmethod
    def is_productfolder(cls, dir_path: Union[str, Path]) -> bool:
        """
        Verify if the folder at the path in input is a ProductFolder

        :param dir_path: path of the folder to check
        :return: True if the folder is a ProductFolder False instead
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            return False

        files = [f.name for f in dir_path.iterdir()]

        # Check manifest existence
        if not cls._get_manifest_file_path(Path(dir_path)).exists():
            return False

        # Exist the raster for each header
        raster_extension = cls.get_data_file_extension(str(dir_path))
        for header in cls._get_header_files_in_dir(dir_path):
            raster_filename = header[0 : -len(_header_extension)] + raster_extension
            if files.count(raster_filename) != 1:
                return False

        # Exist the header for each raster
        for raster_file in cls._get_raster_files_in_dir(dir_path):
            if files.count(raster_file + _header_extension) != 1:
                return False

        return True

    @classmethod
    def get_data_file_extension(cls, dir_path: str) -> str:
        manifest_filename = cls._get_manifest_file_path(Path(dir_path))
        manifest = Manifest.from_file(manifest_filename)
        extension = manifest.datafile_extension.value
        return extension

    def get_number_channels(self) -> int:
        """
        Get the number of channels in the ProductFolder

        :return: number of channels
        """
        return len(self.__channels)

    def read_binary_header(self, channel_index):
        """
        Read the header of the raster of a channel of the ProductFolder

        :param channel_index: index of the channel to read
        :return: header in bytes
        """
        return self.__channels[channel_index].read_binary_header()

    def write_binary_header(self, channel_index, header):
        """
        Write data to a channel of the ProductFolder

        :param channel_index: index of the channel to write
        :param header: header in bytes
        """
        if self._open_mode == channel.EOpenMode.open:
            raise ReadOnlyProductFolder
        self.__channels[channel_index].write_binary_header(header)

    def read_row_prefix(self, channel_index, line_index):
        """
        Read the row prefix of a given line of a channel of the ProductFolder

        :param channel_index: index of the channel
        :param line_index: raster line index
        :return: row prefix as byte sequence
        """
        return self.__channels[channel_index].read_row_prefix(line_index)

    def write_row_prefix(self, channel_index, line_index, row_prefix):
        """
        Write the row prefix of a given line of a channel of the ProductFolder

        :param channel_index: index of the channel
        :param line_index: raster line index
        :param row_prefix: row prefix as bytes sequence
        """
        if self._open_mode == channel.EOpenMode.open:
            raise ReadOnlyProductFolder
        self.__channels[channel_index].write_row_prefix(line_index, row_prefix)

    def read_data(self, channel_index, block_to_read=None):
        """
        Read the raster of a channel of the ProductFolder

        :param channel_index: index of the channel to read
        :param block_to_read: portion of the raster to read [start line, start sample, number of lines, number of
            samples]
        :return: read data
        """
        data = channel.Channel.read_data(self.__channels[channel_index], block_to_read)
        return data

    def write_data(self, channel_index, data, start_point=(0, 0)):
        """
        Write data to a channel of the ProductFolder

        :param channel_index: index of the channel to write
        :param data: data to write
        :param start_point: coordinates of the first pixel to write
        """
        if self._open_mode == channel.EOpenMode.open:
            raise ReadOnlyProductFolder
        self.__channels[channel_index].write_data(data, start_point)

    def write_metadata(self, channel_index):
        """
        Write the metadata to the xml for a channel of the ProductFolder

        :param channel_index: index of the channel to write
        """
        if self._open_mode == channel.EOpenMode.open:
            raise ReadOnlyProductFolder
        self.__channels[channel_index].write_metadata()

    def get_channel(self, channel_index) -> channel.Channel:
        """
        Get a channel of the ProductFolder

        :param channel_index: index of the channel to return
        :return: :class:`arepytools.io.metadata.channel.Channel` -- Channel for the index in input
        """
        return self.__channels[channel_index]


def get_channel_indexes(
    product: ProductFolder,
    swath_name: Optional[str] = None,
    polarization: Optional[Union[EPolarization, str]] = None,
) -> List[int]:
    """
    Get channel indexes that correspond to the swath_name and/or to the polarization requested

    :param product: input product
    :param swath_name: name of the swath
    :param polarization: polarization as an enum or a str
    :return: list of channel indexes

    Examples:
        >>> from arepytools.io.metadata import EPolarization
        >>> get_channel_indexes(pf, swath_name='IW1', polarization=EPolarization.hh)
        >>> get_channel_indexes(pf, swath_name='IW1', polarization='H/H')
        >>> get_channel_indexes(pf, swath_name='IW2')
        >>> get_channel_indexes(pf, polarization=EPolarization.hv)
        >>> get_channel_indexes(pf, polarization='H/V')

    """
    if polarization is not None:
        polarization = EPolarization(polarization)
    number_of_channels = product.get_number_channels()
    channel_indexes = []
    for index_channel in range(number_of_channels):
        channel = product.get_channel(index_channel)

        swath_info = channel.get_swath_info()

        if swath_name is None or swath_info.swath == swath_name:
            if polarization is None or swath_info.polarization == polarization:
                channel_indexes.append(index_channel)

    return channel_indexes


def rename_product_folder(input_product, new_name: str) -> str:
    """Rename an input product with a new name.

    The renamed product is in the same folder as the original product.
    Data, Metadata, Configuration file and manifest are renamed.
    Any additional content in the original folder is kept.

    :param input_product: input product path
    :param new_name: new name of the product
    :return: path to the renamed product

    Examples:
        >>> output_product = rename_product_folder("a/path/to/old_name", "new_name")
        >>> print(output_product)
            'a\\path\\to\\new_name'
    """

    warnings.warn(
        "rename_product_folder function is deprecated starting from Arepytools v1.6.0, "
        + "please use productfolder2.rename_product_folder or the dot method rename of ProductFolder2 class instead",
        RenameProductFolderDeprecationWarning,
        stacklevel=2,
    )

    input_pf = ProductFolder(input_product, "r")

    output_product = Path(input_product).with_name(new_name)
    output_pf = ProductFolder(output_product, "w")

    for channel_index in range(input_pf.get_number_channels()):
        input_channel = input_pf.get_channel(channel_index)

        output_raster_file = input_channel.raster_file.replace(
            input_pf.pf_name, output_pf.pf_name
        )

        Path(input_channel.raster_file).rename(output_raster_file)

        output_metadata_file = input_channel.metadata_file.replace(
            input_pf.pf_name, output_pf.pf_name
        )

        # Create a new channel from the original one
        output_channel = copy.copy(input_channel)
        output_channel._Channel__file_name = Path(output_raster_file).name
        output_channel._raster_file = output_raster_file
        output_channel._metadata_file = output_metadata_file
        output_channel._open_mode = output_pf.open_mode

        # Replace raster name in raster info
        for metadata_channel_index in range(
            output_channel.metadata.get_number_of_channels()
        ):
            raster_info = output_channel.get_raster_info(metadata_channel_index)
            raster_info._file_name = raster_info.file_name.replace(
                input_pf.pf_name, output_pf.pf_name
            )

        output_pf._ProductFolder__channels.append(output_channel)
        output_pf.write_metadata(channel_index)

        # Remove original metadata
        Path(input_channel.metadata_file).unlink()

    # Rename manifest
    Path(output_pf._manifest_file).unlink()
    Path(input_pf._manifest_file).rename(output_pf._manifest_file)

    # Rename config file
    if Path(input_pf.config_file).exists():
        Path(input_pf.config_file).rename(output_pf.config_file)

    # If nothing else is left, remove the original product
    if not list(Path(input_product).iterdir()):
        Path(input_product).rmdir()

    return str(output_product)


def remove_product_folder(product):
    """Remove a product folder

    Only valid products are removed.
    Data, Metadata, Configuration file and manifest only are removed.
    Any additional content in the folder is kept.

    :param product: input product
    """

    warnings.warn(
        "remove_product_folder function is deprecated starting from Arepytools v1.6.0, "
        + "please use productfolder2.delete_product_folder or the dot method rename of ProductFolder2 class instead",
        RenameProductFolderDeprecationWarning,
        stacklevel=2,
    )

    input_pf = ProductFolder(product, "r")

    for channel_index in range(input_pf.get_number_channels()):
        channel = input_pf.get_channel(channel_index)

        Path(channel.raster_file).unlink()
        Path(channel.metadata_file).unlink()

    Path(input_pf._manifest_file).unlink()
    if Path(input_pf.config_file).exists():
        Path(input_pf.config_file).unlink()

    # If nothing else is left, remove the entire folder
    if not list(Path(product).iterdir()):
        Path(product).rmdir()


class IsNotAProductFolder(ValueError):
    """Raised when trying to open a product which is not a valid product folder"""


class ReadOnlyProductFolder(IOError):
    """Raised when attempting to modify a product folder open in read-only mode"""
