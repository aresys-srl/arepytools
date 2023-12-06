# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
**Deprecated** Channel module
-----------------------------

.. deprecated:: v1.6.0
    Using channel module from arepytools.io is deprecated.
"""

import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from arepytools.io import ChannelDeprecationWarning, metadata

from .io_support import (
    get_line_size,
    read_raster_with_raster_info,
    write_metadata,
    write_raster_with_raster_info,
)
from .metadata import MetaData, MetaDataChannel


class EOpenMode(Enum):
    """Initialization modes

    This enumeration contains all the available initialization modalities
    for the ProductFolder object

    """

    #: open a valid product
    open = "r"

    #: create a product, if possible (raises an error if it already exists)
    create = "x"

    #: open a valid product, or, if it not exists, try to create it
    #: (deprecated)
    open_or_create = "a"

    #: create a product, if possible (raises an error if it already exists)
    create_or_overwrite = "w"


# EOpenModeLike = Union[Literal["w", "a", "r", "x"], EOpenMode] # From python 3.8 only
EOpenModeLike = Union[str, EOpenMode]

CHANNEL_ID_RE_PATTERN = "_(?!0000)[0-9]{4}"

DEPRECATION_MSG = "Channel module is deprecated starting from Arepytools v1.6.0"


class DataOutOfRasterInfoLimits(IOError):
    """Raised when attempting to write data out of raster info limits"""


warnings.warn(
    DEPRECATION_MSG,
    ChannelDeprecationWarning,
    stacklevel=2,
)


class Channel:
    """ProductFolder's Channel class"""

    def __init__(self, file_name, metadata, open_mode: EOpenModeLike):
        if not isinstance(metadata, MetaData):
            raise TypeError(f"Unsupported metadata type: {metadata.__class__.__name__}")

        raster_file_path = Path(file_name)
        metadata_file_name = _retrieve_metadata_file_name_from_raster_file_name(
            raster_file_path.name
        )
        metadata_file_path = raster_file_path.parent.joinpath(metadata_file_name)

        open_mode = EOpenMode(open_mode)
        if open_mode == EOpenMode.open:
            if not raster_file_path.is_file():
                raise FileNotFoundError(raster_file_path)

            if not metadata_file_path.is_file():
                raise FileNotFoundError(metadata_file_path)

        self._open_mode = open_mode
        self.__metadata = metadata
        self.__file_name = file_name
        self._raster_file = raster_file_path.absolute()
        self._metadata_file = metadata_file_path.absolute()

        warnings.warn(
            DEPRECATION_MSG,
            ChannelDeprecationWarning,
            stacklevel=2,
        )

    @classmethod
    def from_raster_info(cls, file_name, raster_info, open_mode: EOpenModeLike):
        metadata = MetaData()
        metadata_channel = MetaDataChannel()
        metadata_channel.insert_element(raster_info)
        metadata.append_channel(metadata_channel)
        return cls(file_name, metadata, open_mode)

    def get_sampling_constants(
        self, meta_data_ch_index=0
    ) -> metadata.SamplingConstants:
        """
        Getter of the SamplingConstants

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.SamplingConstants`
        """
        return self.metadata.get_sampling_constants(meta_data_ch_index)

    def get_pulse(self, meta_data_ch_index=0) -> metadata.Pulse:
        """
        Getter of the Pulse

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.Pulse`
        """
        return self.metadata.get_pulse(meta_data_ch_index)

    def get_raster_info(self, meta_data_ch_index=0) -> metadata.RasterInfo:
        """
        Getter of the RasterInfo

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.RasterInfo`
        """
        return self.metadata.get_raster_info(meta_data_ch_index=meta_data_ch_index)

    def get_dataset_info(self, meta_data_ch_index=0) -> metadata.DataSetInfo:
        """
        Getter of the DataSetInfo

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.DataSetInfo`
        """
        return self.metadata.get_dataset_info(meta_data_ch_index)

    def get_state_vectors(self, meta_data_ch_index=0) -> metadata.StateVectors:
        """
        Getter of the StateVectors

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.StateVectors`
        """
        return self.metadata.get_state_vectors(meta_data_ch_index)

    def get_attitude_info(self, meta_data_ch_index=0) -> metadata.AttitudeInfo:
        """
        Getter of the AttitudeInfo

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.AttitudeInfo`
        """
        return self.metadata.get_attitude_info(meta_data_ch_index)

    def get_acquisition_time_line(
        self, meta_data_ch_index=0
    ) -> metadata.AcquisitionTimeLine:
        """
        Getter of the AcquisitionTimeLine

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.AcquisitionTimeLine`
        """
        return self.metadata.get_acquisition_time_line(meta_data_ch_index)

    def get_ground_corner_points(
        self, meta_data_ch_index=0
    ) -> metadata.GroundCornerPoints:
        """
        Getter of the GroundCornerPoints

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.GroundCornerPoints`
        """
        return self.metadata.get_ground_corner_points(meta_data_ch_index)

    def get_burst_info(self, meta_data_ch_index=0) -> metadata.BurstInfo:
        """
        Getter of the BurstInfo

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.BurstInfo`
        """
        return self.metadata.get_burst_info(meta_data_ch_index)

    def get_doppler_centroid(
        self, meta_data_ch_index=0
    ) -> metadata.DopplerCentroidVector:
        """
        Getter of the DopplerCentroidVector

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.DopplerCentroidVector`
        """
        return self.metadata.get_doppler_centroid(meta_data_ch_index)

    def get_doppler_rate(self, meta_data_ch_index=0) -> metadata.DopplerRateVector:
        """
        Getter of the DopplerRateVector

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.DopplerRateVector`
        """
        return self.metadata.get_doppler_rate(meta_data_ch_index)

    def get_tops_azimuth_modulation_rate(
        self, meta_data_ch_index=0
    ) -> metadata.TopsAzimuthModulationRateVector:
        """
        Getter of the TopsAzimuthModulationRateVector

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.TopsAzimuthModulationRateVector`
        """
        return self.metadata.get_tops_azimuth_modulation_rate(meta_data_ch_index)

    def get_slant_to_ground(self, meta_data_ch_index=0) -> metadata.SlantToGroundVector:
        """
        Getter of the SlantToGround

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.SlantToGroundVector`
        """
        return self.metadata.get_slant_to_ground(meta_data_ch_index)

    def get_ground_to_slant(self, meta_data_ch_index=0) -> metadata.GroundToSlantVector:
        """
        Getter of the GroundToSlant

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.GroundToSlantVector`
        """
        return self.metadata.get_ground_to_slant(meta_data_ch_index)

    def get_slant_to_incidence(self, meta_data_ch_index=0) -> metadata.SlantToIncidence:
        """
        Getter of the SlantToIncidence

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.SlantToIncidence`
        """
        return self.metadata.get_slant_to_incidence(meta_data_ch_index)

    def get_slant_to_elevation(self, meta_data_ch_index=0) -> metadata.SlantToElevation:
        """
        Getter of the SlantToElevation

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.SlantToElevation`
        """
        return self.metadata.get_slant_to_elevation(meta_data_ch_index)

    def get_antenna_info(self, meta_data_ch_index=0) -> metadata.AntennaInfo:
        """
        Getter of the AntennaInfo

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.AntennaInfo`
        """
        return self.metadata.get_antenna_info(meta_data_ch_index)

    def get_data_statistics(self, meta_data_ch_index=0) -> metadata.DataStatistics:
        """
        Getter of the DataStatistics

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.DataStatistics`
        """
        return self.metadata.get_data_statistics(meta_data_ch_index)

    def get_swath_info(self, meta_data_ch_index=0) -> metadata.SwathInfo:
        """
        Getter of the SwathInfo

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.SwathInfo`
        """
        return self.metadata.get_swath_info(meta_data_ch_index)

    def get_coreg_poly(self, meta_data_ch_index=0) -> metadata.CoregPolyVector:
        """
        Getter of the CoregPolyVector

        :param meta_data_ch_index: index of the metadata channel
        :return: :class:`arepytools.io.metadata.CoregPolyVector`
        """
        return self.metadata.get_coreg_poly(meta_data_ch_index)

    @property
    def metadata(self) -> MetaData:
        return self.__metadata

    @property
    def open_mode(self):
        return self._open_mode

    @property
    def file_name(self):
        """
        .. deprecated:: v1.1.0

            Use :data:`raster_file` property instead.
        """
        warnings.warn(
            "file_name property is deprecated: use raster_file property instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.__file_name

    @property
    def raster_file(self):
        """Raster file absolute path"""
        return str(self._raster_file)

    @property
    def metadata_file(self):
        """Metadata file absolute path"""
        return str(self._metadata_file)

    def read_data(self, block_to_read=None):
        """
        Read the raster

        :param block_to_read: portion of the raster to read [start line, start sample, number of lines, number of
            samples]
        :return: data, a matrix with shape (lines, samples)
        """

        # Get RasterInfo from metadata
        raster_info = self.get_raster_info()
        data = read_raster_with_raster_info(
            raster_file=self._raster_file,
            raster_info=raster_info,
            block_to_read=block_to_read,
        )

        return data

    def read_binary_header(self):
        """
        Read the data header

        :return: data header in bytes
        """
        raster_info = self.get_raster_info()
        if raster_info.header_offset_bytes == 0:
            return b""

        with open(self._raster_file, "rb") as raster_file:
            header = raster_file.read(raster_info.header_offset_bytes)

        return header

    def write_binary_header(self, header):
        """
        Write the data header

        :param header: header in bytes
        """
        if self._open_mode == EOpenMode.open:
            raise ReadOnlyChannel

        raster_info = self.get_raster_info()
        header_size = len(header)
        if raster_info.header_offset_bytes != header_size:
            raise RuntimeError(
                "Header size incompatible with header offset: {} != {}".format(
                    header_size, raster_info.header_offset_bytes
                )
            )

        with open(self._raster_file, "wb") as raster_file:
            raster_file.write(header)

    def read_row_prefix(self, line_index):
        """
        Read the row prefix of a given line.

        :param line_index: raster line index
        :return: row prefix as byte sequence
        """
        raster_info = self.get_raster_info()
        if raster_info.row_prefix_bytes == 0:
            return b""

        offset_byte = raster_info.header_offset_bytes + line_index * get_line_size(
            raster_info.samples, raster_info.cell_type, raster_info.row_prefix_bytes
        )

        with open(self._raster_file, "rb") as raster_file:
            raster_file.seek(offset_byte)
            row_prefix = raster_file.read(raster_info.row_prefix_bytes)

        return row_prefix

    def write_row_prefix(self, line_index, row_prefix):
        """
        Write the row prefix at a given line.

        :param line_index: raster line index
        :param row_prefix: row prefix as byte sequence
        """
        if self._open_mode == EOpenMode.open:
            raise ReadOnlyChannel

        raster_info = self.get_raster_info()
        row_prefix_size = len(row_prefix)
        if raster_info.row_prefix_bytes != row_prefix_size:
            raise RuntimeError(
                "Row prefix size incompatible with row prefix size in raster info: {} != {}".format(
                    row_prefix_size, raster_info.row_prefix_bytes
                )
            )

        offset_byte = raster_info.header_offset_bytes + line_index * get_line_size(
            raster_info.samples, raster_info.cell_type, raster_info.row_prefix_bytes
        )

        with open(self._raster_file, "wb") as raster_file:
            raster_file.seek(offset_byte)
            raster_file.write(row_prefix)

    def write_data(self, data, start_point=(0, 0)):
        """
        Write the raster

        :param data:
        :param start_point:
        """
        if self._open_mode == EOpenMode.open:
            raise ReadOnlyChannel

        # Get RasterInfo from metadata
        raster_info = self.get_raster_info()
        write_raster_with_raster_info(
            raster_file=self._raster_file,
            data=data,
            raster_info=raster_info,
            start_point=start_point,
        )

    def write_metadata(self):
        """
        Write metadata to xml
        """
        if self._open_mode == EOpenMode.open:
            raise ReadOnlyChannel
        write_metadata(self.metadata, self.metadata_file)


class ReadOnlyChannel(IOError):
    """Raised when attempting to write in a channel open in read-only mode"""


def _retrieve_extension_from_raster_file_name(raster_file_name: str) -> Optional[str]:
    """Given a raster file name, retrieve the extension if any

    Parameters
    ----------
    raster_file_name : str
        name of the raster

    Returns
    -------
    Optional[str]
        extension if present

    Raises
    ------
    RuntimeError
        raised when the name does not match the channel id pattern
    """
    name_split = re.split(CHANNEL_ID_RE_PATTERN, raster_file_name)
    if len(name_split) < 2:
        raise RuntimeError(f"Unexpected raster file name: {raster_file_name}")

    extension = name_split[-1]

    return extension if extension else None


def _retrieve_metadata_file_name_from_raster_file_name(raster_file_name: str) -> str:
    """Get the metadata file name corresponding to the given raster file

    Parameters
    ----------
    raster_file_name : str
        name of the raster file

    Returns
    -------
    str
        corresponding metadata file name
    """
    raster_extension = _retrieve_extension_from_raster_file_name(raster_file_name)

    if raster_extension is not None:
        return raster_file_name.replace(raster_extension, ".xml")
    else:
        return raster_file_name + ".xml"
