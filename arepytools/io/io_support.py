# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
IO support module
-----------------
"""

import os.path
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from arepytools.io import metadata
from arepytools.io.parsing.metadata_parsing import parse_metadata, serialize_metadata


class DataOutOfRasterInfoLimits(IOError):
    """Raised when attempting to write data out of raster info limits"""


data_type_dict = {
    "FLOAT32": "f4",
    "FLOAT_COMPLEX": "c8",
    "INT16": "i2",
    "INT32": "i4",
    "UINT16": "u2",
    "UINT32": "u4",
    "INT8": "b",
    "UINT8": "B",
    "INT16_COMPLEX": "i2, i2",
    "DOUBLE_COMPLEX": "c16",
    "FLOAT64": "f8",
    "INT8_COMPLEX": "i1, i1",
}

byte_order_dict = {"BIGENDIAN": ">", "LITTLEENDIAN": "<"}

_UNSUPPORTED_TYPES = ("INT8_COMPLEX", "INT16_COMPLEX")


class UnsupportedDataType(RuntimeError):
    """Specified data type is not supported"""


class InvalidDataType(RuntimeError):
    """Invalid data type"""


class BlockExceedsRasterLimits(RuntimeError):
    """Block to be read exceeds raster limits"""


class InvalidRowPrefix(ValueError):
    """Invalid prefix value"""


class InvalidHeaderOffset(ValueError):
    """Invalid header offset"""


class InvalidWritingPoint(ValueError):
    """Invalid writing point"""


class InvalidHeaderSize(RuntimeError):
    """Header size to be written is not compatible with header offset in raster file"""


class InvalidRowPrefixSize(RuntimeError):
    """Row prefix size to be written is not compatible with row prefix size in raster file"""


class InvalidPointTargetError(RuntimeError):
    """Invalid Point Target file or binary path"""


class PointTargetDimensionsMismatchError(RuntimeError):
    """Invalid matching between Point Target input values when writing a Point Target file"""


class InvalidRasterBlockError(RuntimeError):
    """Required raster block to be read is not valid"""


class CoordinatesType(Enum):
    """Enum class for nominal point target coordinates type"""

    LLH = 0
    ECEF = 1
    NORMALIZED = 2


class OpenMode(Enum):
    """Enum class for opening mode of PointSetProduct"""

    READ = "r"
    WRITE = "w"  # write and overwrite


@dataclass
class NominalPointTarget:
    """Nominal Point Target dataclass containing info gathered from xml file"""

    xyz_coordinates: np.ndarray = None
    rcs_hh: np.complex128 = None
    rcs_vv: np.complex128 = None
    rcs_vh: np.complex128 = None
    rcs_hv: np.complex128 = None
    delay: float = None


def read_raster_with_raster_info(
    raster_file: Union[str, Path],
    raster_info: metadata.RasterInfo,
    block_to_read: List[int] = None,
) -> np.ndarray:
    """Read raster file using information from a RasterInfo metadata object.

    Parameters
    ----------
    raster_file : Union[str, Path]
        path to the raster file to be read
    raster_info : metadata.RasterInfo
        RasterInfo metadata corresponding to the raster to be read
    block_to_read : List[int], optional
        data block to be read, to be specified as a list of 4 integers, in the form:
            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """
    raster_file = Path(raster_file)

    if raster_file.name != raster_info.file_name:
        warnings.warn(
            f"Raster file name {raster_file.name} differs from raster info file name {raster_info.file_name}"
        )

    return read_raster(
        raster_file_name=raster_file,
        num_of_samples=raster_info.samples,
        num_of_lines=raster_info.lines,
        data_type=raster_info.cell_type,
        binary_ordering_mode=raster_info.byte_order,
        block_to_read=block_to_read,
        header_offset=raster_info.header_offset_bytes,
        row_prefix=raster_info.row_prefix_bytes,
    )


def read_raster(
    raster_file_name: Union[str, Path],
    num_of_samples: int,
    num_of_lines: int,
    data_type: Union[str, metadata.ECellType] = metadata.ECellType.float32,
    binary_ordering_mode: Union[str, metadata.EByteOrder] = metadata.EByteOrder.le,
    block_to_read: List[int] = None,
    header_offset: int = 0,
    row_prefix: int = 0,
) -> np.ndarray:
    """Read raster file data.

    Parameters
    ----------
    raster_file_name : Union[str, Path]
        path to the raster file to be read
    num_of_samples : int
        number of samples to be read
    num_of_lines : int
        number of lines to be read
    data_type : Union[str, metadata.ECellType], optional
        data type corresponding to the raster itself, by default metadata.ECellType.float32
    binary_ordering_mode : Union[str, metadata.EByteOrder], optional
        binary ordering mode corresponding to the raster itself, by default metadata.EByteOrder.le
    block_to_read : List[int], optional
        data block to be read, to be specified as a list of 4 integers, in the form:
            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        if None, the whole raster is read, by default None
    header_offset : int, optional
        header offset of the raster file, by default 0
    row_prefix : int, optional
        row prefix of the raster file, by default 0

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)

    Raises
    ------
    InvalidHeaderOffset
        if header offset is negative
    InvalidRowPrefix
        if row prefix is negative
    UnsupportedDataType
        invalid data type
    BlockExceedsRasterLimits
        data type not yet supported
    BlockExceedsRasterLimits
        if first line to be read is negative
    BlockExceedsRasterLimits
        if first sample to be read is negative
    BlockExceedsRasterLimits
        if block to be read exceeds raster number of lines
    BlockExceedsRasterLimits
        if block to be read exceeds raster number of samples
    """

    data_type = metadata.ECellType(data_type)
    binary_ordering_mode = metadata.EByteOrder(binary_ordering_mode)

    if header_offset < 0:
        raise InvalidHeaderOffset("header_offset should be non-negative")

    if row_prefix < 0:
        raise InvalidRowPrefix("row_prefix should be non-negative")

    if data_type.value in _UNSUPPORTED_TYPES:
        raise UnsupportedDataType(
            f"Read data from raster of type {data_type.value} currently not supported."
        )

    if data_type.value in data_type_dict:
        data_type_numpy_value = data_type_dict[data_type.value]
        data_type = np.dtype(data_type_dict[data_type.value])
    else:
        raise InvalidDataType(f"Unknown data type id: {data_type.value}")

    file_data_type = np.dtype(
        byte_order_dict[binary_ordering_mode.value] + data_type_numpy_value
    )

    # Compute the items to read
    if block_to_read is None:
        lines_to_read = num_of_lines
        samples_to_read = num_of_samples
        first_line = 0
        first_sample = 0
    else:
        lines_to_read = block_to_read[2]
        samples_to_read = block_to_read[3]
        first_line = block_to_read[0]
        first_sample = block_to_read[1]

    # convert to int
    lines_to_read = int(lines_to_read)
    samples_to_read = int(samples_to_read)
    first_line = int(first_line)
    first_sample = int(first_sample)

    if first_line < 0:
        raise BlockExceedsRasterLimits("First line to read should be non-negative")

    if first_sample < 0:
        raise BlockExceedsRasterLimits("First sample to read should be non-negative")

    if first_line + lines_to_read > num_of_lines:
        raise BlockExceedsRasterLimits("Block to read exceeds max num lines")

    if first_sample + samples_to_read > num_of_samples:
        raise BlockExceedsRasterLimits("Block to read exceeds max num samples")

    # Read data from file
    with open(raster_file_name, "rb") as fdesc:
        if samples_to_read == num_of_samples and row_prefix == 0:
            offset_byte = (
                header_offset + first_line * num_of_samples * data_type.itemsize
            )
            data = np.fromfile(
                fdesc,
                dtype=file_data_type,
                count=lines_to_read * samples_to_read,
                offset=offset_byte,
            )

            return data.reshape((lines_to_read, samples_to_read))

        data = np.empty((lines_to_read, samples_to_read), dtype=data_type)

        offset_byte = (
            (first_line * num_of_samples + first_sample) * data_type.itemsize
            + row_prefix * (first_line + 1)
            + header_offset
        )
        fdesc.seek(offset_byte, 0)

        offset_line_byte = (
            num_of_samples - samples_to_read
        ) * data_type.itemsize + row_prefix

        for line in range(lines_to_read):
            offset_byte = offset_line_byte if line > 0 else 0
            data[line, :] = np.fromfile(
                fdesc, dtype=file_data_type, count=samples_to_read, offset=offset_byte
            )

        return data


def write_raster_with_raster_info(
    raster_file: Union[str, Path],
    data: np.ndarray,
    raster_info: metadata.RasterInfo,
    start_point: Tuple[int, int] = (0, 0),
) -> None:
    """Write data to the specified raster file on disk.

    Parameters
    ----------
    raster_file : Union[str, Path]
        path to the raster file to be written
    data : np.ndarray
        data to be written
    raster_info : metadata.RasterInfo
        RasterInfo object containing all the info related to the raster of choice
    start_point : Tuple[int, int], optional
        start point from where to write (lines, samples), by default (0, 0)

    Raises
    ------
    DataOutOfRasterInfoLimits
        when trying to write data out of raster limits
    """
    raster_file = Path(raster_file)

    if raster_file.name != raster_info.file_name:
        warnings.warn(
            f"Raster file name {raster_file.name} differs from raster info file name {raster_info.file_name}"
        )

    max_lines = data.shape[0] + start_point[0]
    max_samples = data.shape[1] + start_point[1]
    if max_lines <= raster_info.lines and max_samples <= raster_info.samples:
        write_raster(
            raster_file_name=raster_file,
            data=data,
            num_of_samples=raster_info.samples,
            num_of_lines=raster_info.lines,
            data_type=raster_info.cell_type,
            binary_ordering_mode=raster_info.byte_order,
            writing_point=start_point,
            header_offset=raster_info.header_offset_bytes,
            row_prefix=raster_info.row_prefix_bytes,
        )
    else:
        raise DataOutOfRasterInfoLimits


def write_raster(
    raster_file_name: Union[str, Path],
    data: np.ndarray,
    num_of_samples: int,
    num_of_lines: int,
    data_type: Union[str, metadata.ECellType] = metadata.ECellType.float32,
    binary_ordering_mode: Union[str, metadata.EByteOrder] = metadata.EByteOrder.le,
    writing_point: Tuple[int, int] = (0, 0),
    header_offset: int = 0,
    row_prefix: int = 0,
) -> None:
    """Write raster file to disk.

    Parameters
    ----------
    raster_file_name : Union[str, Path]
        path to the raster file to be read
    data : np.ndarray
        data to be written
    num_of_samples : int
        number of samples to be read
    num_of_lines : int
        number of lines to be read
    data_type : Union[str, metadata.ECellType], optional
        data type corresponding to the raster itself, by default metadata.ECellType.float32
    binary_ordering_mode : Union[str, metadata.EByteOrder], optional
        binary ordering mode corresponding to the raster itself, by default metadata.EByteOrder.le
    writing_point : Tuple[int, int], optional
        line and sample from where to start writing data, by default (0, 0)
    header_offset : int, optional
        header offset of the raster file, by default 0
    row_prefix : int, optional
        row prefix of the raster file, by default 0

    Raises
    ------
    InvalidHeaderOffset
        if header offset is negative
    InvalidRowPrefix
        if row prefix is negative
    ValueError
        if writing point has more than 2 elements or negative values
    ValueError
        data type not yet supported
    ValueError
        if block to be written exceeds raster number of samples
    ValueError
        if block to be read exceeds raster number of lines
    RuntimeError
        unexpected file size
    RuntimeError
        unexpected final raster size
    """

    data_type = metadata.ECellType(data_type)
    binary_ordering_mode = metadata.EByteOrder(binary_ordering_mode)

    if header_offset < 0:
        raise InvalidHeaderOffset("header_offset should be non-negative")

    if row_prefix < 0:
        raise InvalidRowPrefix("row_prefix should be non-negative")

    if len(writing_point) != 2 or writing_point[0] < 0 or writing_point[1] < 0:
        raise InvalidWritingPoint("Writing point should have two non-negative elements")

    if data_type.value in _UNSUPPORTED_TYPES:
        raise UnsupportedDataType(
            f"Write data to raster of type {data_type.value} currently not supported."
        )

    if data_type.value in data_type_dict:
        data_type_numpy_value = data_type_dict[data_type.value]
        data_type = np.dtype(data_type_dict[data_type.value])
    else:
        raise InvalidDataType(f"Unknown data type id: {data_type.value}")

    file_data_type = np.dtype(
        byte_order_dict[binary_ordering_mode.value] + data_type_numpy_value
    )

    # Convert data to data type
    data_to_write = np.array(data, dtype=file_data_type)

    first_line, first_sample = writing_point
    lines_to_write, samples_to_write = data_to_write.shape

    if first_sample + samples_to_write > num_of_samples:
        raise ValueError("Input data exceeds max num samples")

    if first_line + lines_to_write > num_of_lines:
        raise ValueError("Input data exceeds max num lines")

    if os.path.isfile(raster_file_name):
        open_mode = "r+b"
    else:
        open_mode = "wb"

    with open(raster_file_name, open_mode) as fdesc:
        raster_size = int(
            header_offset
            + row_prefix * num_of_lines
            + data_type.itemsize * num_of_samples * num_of_lines
        )

        # check raster has the correct size
        fdesc.seek(0, 2)
        file_size = fdesc.tell()
        if file_size < raster_size:
            last_element_position = raster_size - data_type.itemsize
            fdesc.seek(last_element_position)
            fdesc.write(np.array(0, dtype=data_type).tobytes())
        fdesc.seek(0, 2)
        file_size = fdesc.tell()
        if file_size != raster_size:
            raise RuntimeError("Unexpected file size")

        for line in range(0, lines_to_write):
            # Move file cursor to correct position
            absolute_line = line + first_line
            absolute_position = first_sample + num_of_samples * absolute_line
            past_row_prefix_size = row_prefix * (line + first_line + 1)
            write_position = int(
                absolute_position * data_type.itemsize
                + header_offset
                + past_row_prefix_size
            )
            fdesc.seek(write_position)

            # Write
            line_to_write = data_to_write[line].tobytes()
            fdesc.write(line_to_write)

        fdesc.seek(0, 2)
        if fdesc.tell() != raster_size:
            raise RuntimeError("Unexpected final raster size")


def read_metadata(metadata_file: Union[str, Path]) -> metadata.MetaData:
    """Read metadata from XML file

    Parameters
    ----------
    metadata_file : Union[str, Path]
        path to the xml aresys metadata file

    Returns
    -------
    metadata.MetaData
        Channel metadata
    """
    metadata_content = Path(metadata_file).read_text(encoding="utf-8")
    return parse_metadata(metadata_content)


def write_metadata(
    metadata_obj: metadata.MetaData, metadata_file: Union[str, Path]
) -> None:
    """Write metadata to XML file

    Parameters
    ----------
    metadata_obj : metadata.Metadata
        Channel metadata

    metadata_file : Path
        path to the xml aresys metadata file
    """
    metadata_content = serialize_metadata(metadata_obj)
    Path(metadata_file).write_text(metadata_content, encoding="utf-8")


def create_new_metadata(
    num_metadata_channels: int = 1, description: str = None
) -> metadata.MetaData:
    """Create a new empty MetaData object with the selected number of metadata channels.

    Parameters
    ----------
    num_metadata_channels : int, optional
        number of metadata channels, by default 1
    description : str, optional
        metadata description, by default None

    Returns
    -------
    metadata.MetaData
        new empty MetaData object
    """
    if description is None:
        description = ""

    meta_data = metadata.MetaData(description=description)
    for _ in range(num_metadata_channels):
        meta_data.append_channel(metadata.MetaDataChannel())

    return meta_data


def read_binary_header_with_raster_info(
    raster_file: Union[str, Path], raster_info: metadata.RasterInfo
) -> bytes:
    """Read raster binary header using information from a RasterInfo metadata object.

    Parameters
    ----------
    raster_file : Union[str, Path]
        path to the raster file to be read
    raster_info : metadata.RasterInfo
        RasterInfo metadata corresponding to the raster to be read

    Returns
    -------
    bytes
        header binary of the raster file
    """
    raster_file = Path(raster_file)

    if raster_info.header_offset_bytes == 0:
        return b""

    with open(raster_file, "rb") as rf:
        header = rf.read(raster_info.header_offset_bytes)

    return header


def write_binary_header_with_raster_info(
    raster_file: Union[str, Path], header: bytes, raster_info: metadata.RasterInfo
):
    """Write raster binary header using information from a RasterInfo metadata object.

    Parameters
    ----------
    raster_file : Union[str, Path]
        path to the raster file to be read
    header : bytes
        header to be written in bytes
    raster_info : metadata.RasterInfo
        RasterInfo metadata corresponding to the raster to be read

    Raises
    ------
    InvalidHeaderOffset
        header size incompatible with header offset
    """
    raster_file = Path(raster_file)

    header_size = len(header)
    offset_size = raster_info.header_offset_bytes
    if offset_size != header_size:
        raise InvalidHeaderOffset(
            f"Header size incompatible with header offset: {header_size} != {offset_size}"
        )

    with open(raster_file, "wb") as rf:
        rf.write(header)


def read_row_prefix_with_raster_info(
    raster_file: Union[str, Path], line_index: int, raster_info: metadata.RasterInfo
) -> bytes:
    """Read the row prefix of a given line using information from a RasterInfo metadata object.

    Parameters
    ----------
    raster_file : Union[str, Path]
        path to the raster file to be read
    line_index : int
        raster line index
    raster_info : metadata.RasterInfo
        RasterInfo metadata corresponding to the raster to be read

    Returns
    -------
    bytes
        row prefix as byte sequence
    """
    raster_file = Path(raster_file)

    if raster_info.row_prefix_bytes == 0:
        return b""

    offset_byte = raster_info.header_offset_bytes + line_index * get_line_size(
        raster_info.samples, raster_info.cell_type, raster_info.row_prefix_bytes
    )

    with open(raster_file, "rb") as rf:
        rf.seek(offset_byte)
        row_prefix = rf.read(raster_info.row_prefix_bytes)

    return row_prefix


def write_row_prefix_with_raster_info(
    raster_file: Union[str, Path],
    line_index: int,
    row_prefix: bytes,
    raster_info: metadata.RasterInfo,
):
    """Write the row prefix at a given line using information from a RasterInfo metadata object.

    Parameters
    ----------
    raster_file : Union[str, Path]
        path to the raster file to be read
    line_index : int
        raster line index
    row_prefix : bytes
        row prefix as byte sequence
    raster_info : metadata.RasterInfo
        RasterInfo metadata corresponding to the raster to be read

    Raises
    ------
    InvalidRowPrefixSize
        row prefix size incompatible with row prefix size in raster info
    """
    raster_file = Path(raster_file)

    row_prefix_size = len(row_prefix)
    row_prefix_size_ri = raster_info.row_prefix_bytes
    if row_prefix_size_ri != row_prefix_size:
        raise InvalidRowPrefixSize(
            "Row prefix size incompatible with row prefix size in raster info:"
            + f"{row_prefix_size} != {row_prefix_size_ri}"
        )

    offset_byte = raster_info.header_offset_bytes + line_index * get_line_size(
        raster_info.samples, raster_info.cell_type, row_prefix_size_ri
    )

    with open(raster_file, "wb") as rf:
        rf.seek(offset_byte)
        rf.write(row_prefix)


def get_line_size(
    samples: int, cell_type: metadata.ECellType, row_prefix_size: int
) -> int:
    """Get the size in bytes of a line in the raster (including row prefix).

    Parameters
    ----------
    samples : int
        number of samples per line
    cell_type : metadata.ECellType
        data type
    row_prefix_size : int
        row prefix size in bytes

    Returns
    -------
    int
        line size in bytes
    """
    data_type = np.dtype(data_type_dict[cell_type.value])
    return samples * data_type.itemsize + row_prefix_size
