# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Point Target Binary Module
--------------------------
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

import arepytools.io.io_support as io_utils
import arepytools.io.metadata as mtd

_COORDINATES_RASTER_FILENAMES = [
    "PointTargetPosX",
    "PointTargetPosY",
    "PointTargetPosZ",
]

_RCS_RASTER_FILENAMES = [
    "PointTargetRCSHH",
    "PointTargetRCSHV",
    "PointTargetRCSVH",
    "PointTargetRCSVV",
]

METADATA_EXTENSION = ".xml"


class PointSetProduct:
    """PointSetProduct class representing the point target binary file"""

    def __init__(
        self,
        path: Union[str, Path],
        open_mode: Union[str, io_utils.OpenMode] = io_utils.OpenMode.READ,
    ) -> None:
        """PointSetProduct init.

        Parameters
        ----------
        path : Union[str, Path]
            path to the product folder
        open_mode : Union[str, io_utils.OpenMode], optional
            open mode (can be write "w" or read "r"), write mode will overwrite existing data,
            by default io_utils.OpenMode.READ

        Raises
        ------
        io_utils.InvalidPointTargetError
            if in read mode, folder does not exist
        io_utils.InvalidPointTargetError
            if in read mode, path is not a directory
        """
        self._path = Path(path)
        self._open_mode = io_utils.OpenMode(open_mode)

        # generating full paths for metadata and raster files
        self._coords_metadata_files = [
            self._path.joinpath(c + METADATA_EXTENSION)
            for c in _COORDINATES_RASTER_FILENAMES
        ]
        self._coords_raster_files = [
            self._path.joinpath(c) for c in _COORDINATES_RASTER_FILENAMES
        ]
        self._rcs_metadata_files = [
            self._path.joinpath(r + METADATA_EXTENSION) for r in _RCS_RASTER_FILENAMES
        ]
        self._rcs_raster_files = [self._path.joinpath(r) for r in _RCS_RASTER_FILENAMES]

        if self._open_mode == io_utils.OpenMode.READ:
            # reading mode, asserting existence and being a directory
            if not self._path.exists():
                raise io_utils.InvalidPointTargetError(
                    f"Path does not exist {self._path}"
                )
            if not self._path.is_dir():
                raise io_utils.InvalidPointTargetError(
                    f"Path is not a directory {self._path}"
                )
            # reading number of targets from files
            self._raster_infos, self._num_targets = self._read_num_lines()

            # 3 coordinates files + 4 rcs polarizations
            assert len(self._coords_raster_files) + len(self._rcs_raster_files) == 7
        else:
            # writing mode
            self._raster_infos = None
            self._num_targets = 0

    @property
    def product_path(self) -> Path:
        """Get PointSetProduct folder path"""
        return self._path

    @property
    def number_of_targets(self) -> int:
        """Get number of targets inside the PointSetProduct folder"""
        return self._num_targets

    def _read_num_lines(self) -> Tuple[List[mtd.RasterInfo], int]:
        """Reading the metadata files inside the product folder to extract the number of lines, i.e. the number of
        targets and full raster info for each metadata.
        Checking also that all metadata files are consistent regarding number of lines value.

        Returns
        -------
        Tuple[List[mtd.RasterInfo], int]
            list of raster info for each metadata,
            number of point targets in the product folder binary

        Raises
        ------
        RuntimeError
            number of samples is different from 1
        """

        metadata = self._coords_metadata_files + self._rcs_metadata_files
        lines = []
        raster_infos = []
        for file in metadata:
            raster_info = io_utils.read_metadata(file).get_raster_info()
            lines.append(raster_info.lines)
            raster_infos.append(raster_info)
            if raster_info.samples != 1:
                raise RuntimeError("Number of samples is not 1")

        assert (
            len(set(lines)) == 1
        ), "Metadata files are not consistent: different numbers of lines"
        return raster_infos, lines[0]

    def _read_rasters(self, start: int, stop: int) -> Tuple[np.ndarray, np.ndarray]:
        """Reading binary raster files both for coordinates and rcs.

        Parameters
        ----------
        start : int
            start reading block
        stop : int
            last reading block

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            coordinates array (N, 3),
            rcs array (N, 4)
        """

        # block to be read [first line, first sample, lines to be read, samples to be read]
        block = [start, 0, stop - start, 1]

        assert stop <= self._num_targets
        assert start >= 0

        data = []
        for index, file in enumerate(
            self._coords_raster_files + self._rcs_raster_files
        ):
            data.append(
                io_utils.read_raster_with_raster_info(
                    raster_file=file,
                    raster_info=self._raster_infos[index],
                    block_to_read=block,
                )
            )

        return np.hstack(data[:3]), np.hstack(data[-4:])

    @staticmethod
    def _write_rasters(
        data: np.ndarray, filenames: List[Path], data_type: mtd.ECellType
    ) -> None:
        """Writing input data to raster files using the specified data type.

        Parameters
        ----------
        data : np.ndarray
            array to be written to raster file
        filenames : List[Path]
            names of raster files to be written
        data_type : mtd.ECellType
            data type to be used when writing data
        """
        for index, file in enumerate(filenames):
            data_slice = np.atleast_2d(data[index]).T
            io_utils.write_raster(
                raster_file_name=file,
                data=data_slice,
                num_of_samples=1,
                num_of_lines=data_slice.size,
                data_type=data_type,
            )

    @staticmethod
    def _write_metadata(
        filenames: List[Path], num_points: int, data_type: mtd.ECellType
    ) -> None:
        """Writing metadata files corresponding to raster files.

        Parameters
        ----------
        filenames : List[Path]
            metadata filenames to be written
        num_points : int
            total number of data blocks written to the corresponding raster file
        data_type : mtd.ECellType
            data type used in writing the corresponding raster file
        """
        for file in filenames:
            raster_info = mtd.RasterInfo(
                lines=num_points,
                samples=1,
                celltype=data_type.value,
                filename=file.stem,
            )

            metadata = io_utils.create_new_metadata(
                description="Aresys XML metadata file"
            )
            metadata.insert_element(raster_info)

            io_utils.write_metadata(metadata, str(file))

    def read_data(
        self, start: int = 0, num_points: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reading Point Target Binary rasters to extract point target data.

        Parameters
        ----------
        start : int, optional
            block number (i.e. number of point target) from which start reading the raster, by default 0
        num_points : int, optional
            number of points to be read, if None all points are read from the selected start to the end,
            by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            coordinates array in the form (N, 3),
            rcs array (HH, HV, VV, VH) in the form (N, 4)

        Raises
        ------
        io_utils.InvalidRasterBlockError
            if start is a negative number
        io_utils.InvalidRasterBlockError
            if num_points is a negative number or 0
        io_utils.InvalidRasterBlockError
            if start + num_points exceeds total number of readable blocks
        io_utils.InvalidRasterBlockError
            if start exceeds last readable block
        io_utils.InvalidRasterBlockError
            if num_points is greater than total number of blocks
        """

        if num_points is None:
            num_points = self._num_targets - start

        stop = start + num_points

        if start < 0:
            raise io_utils.InvalidRasterBlockError(
                f"Starting block cannot be negative {start}"
            )

        if num_points <= 0:
            raise io_utils.InvalidRasterBlockError(
                f"Number of blocks to be read cannot be 0 or negative {num_points}"
            )

        if stop > self._num_targets:
            raise io_utils.InvalidRasterBlockError(
                "Blocks to be read exceed total number of readable blocks: "
                + f"start + points to be read {stop} > "
                + f"total number of points {self._num_targets}"
            )

        if start > self._num_targets:
            raise io_utils.InvalidRasterBlockError(
                "Starting block exceeds total number of blocks: "
                + f"{start} > {self._num_targets}"
            )

        if num_points > self._num_targets:
            raise io_utils.InvalidRasterBlockError(
                "Number of blocks to be read exceeds total number of blocks: "
                + f"{num_points} > {self._num_targets}"
            )

        return self._read_rasters(start=start, stop=stop)

    def write_data(
        self,
        coords: np.ndarray,
        rcs: np.ndarray,
        coords_data_type: Union[str, mtd.ECellType] = mtd.ECellType.float64,
        rcs_data_type: Union[str, mtd.ECellType] = mtd.ECellType.fcomplex,
    ) -> None:
        """Writing data to the Point Set Target folder.

        Parameters
        ----------
        coords : np.ndarray
            point target coordinates, in the form (N, 3)
        rcs : np.ndarray
            point target radar cross section values, in the form (N, 4)
        coords_data_type : Union[str, mtd.ECellType], optional
            data type to be used in writing coordinates rasters, by default mtd.ECellType.float64
        rcs_data_type : Union[str, mtd.ECellType], optional
            data type to be used in writing rcs rasters, by default mtd.ECellType.fcomplex

        Raises
        ------
        RuntimeError
            if coordinates shape is wrong
        RuntimeError
            if rcs shape is wrong
        io_utils.PointTargetDimensionsMismatchError
            if coordinates shape does not match rcs shape
        """

        coords = np.atleast_2d(coords)
        rcs = np.atleast_2d(rcs)

        if coords.shape[1] != 3:
            raise RuntimeError(f"Wrong shape: {coords.shape[1]} != 3")

        if rcs.shape[1] != 4:
            raise RuntimeError(f"Wrong shape: {rcs.shape[1]} != 4")

        num_points = coords.shape[0]

        if num_points != rcs.shape[0]:
            raise io_utils.PointTargetDimensionsMismatchError(
                f"number of coordinates {num_points} != number of rcs {rcs.shape[0]}"
            )

        self._path.mkdir(exist_ok=True)

        coords_data_type = mtd.ECellType(coords_data_type)
        rcs_data_type = mtd.ECellType(rcs_data_type)

        # writing coordinates and rcs raster data
        self._write_rasters(
            data=coords.T,
            filenames=self._coords_raster_files,
            data_type=coords_data_type,
        )
        self._write_rasters(
            data=rcs.T, filenames=self._rcs_raster_files, data_type=rcs_data_type
        )

        # writing coordinates and rcs raster metadata
        self._write_metadata(
            filenames=self._coords_metadata_files,
            num_points=num_points,
            data_type=coords_data_type,
        )
        self._write_metadata(
            filenames=self._rcs_metadata_files,
            num_points=num_points,
            data_type=rcs_data_type,
        )


def convert_array_to_point_target_structure(
    coords: np.ndarray, rcs: np.ndarray, point_target_ids: List[str] = None
) -> Dict[str, io_utils.NominalPointTarget]:
    """Converting coordinates and rcs arrays to an array of structures, i.e. a dictionary
    of NominalPointTarget dataclasses each one representing a single point target object.

    Parameters
    ----------
    coords : np.ndarray
        point target coordinates, in the form (N, 3)
    rcs : np.ndarray
        point target rcs values (HH, HV, VV, VH), in the form (N, 4)
    point_target_ids : List[str], optional
        optional list of point target id labels, by default None

    Returns
    -------
    Dict[str, io_utils.NominalPointTarget]
        keys are the target ID, values are the corresponding NominalPointTarget objects

    Raises
    ------
    RuntimeError
        if coordinates shape is wrong
    RuntimeError
        if rcs shape is wrong
    io_utils.PointTargetDimensionsMismatchError
        if coordinates shape does not match rcs shape
    io_utils.PointTargetDimensionsMismatchError
        if point targets ids shape does not match coordinates shape
    """

    coords = np.atleast_2d(coords)
    rcs = np.atleast_2d(rcs)

    if coords.shape[1] != 3:
        raise RuntimeError(f"Wrong shape: {coords.shape[1]} != 3")

    if rcs.shape[1] != 4:
        raise RuntimeError(f"Wrong shape: {rcs.shape[1]} != 4")

    if coords.shape[0] != rcs.shape[0]:
        raise io_utils.PointTargetDimensionsMismatchError(
            f"number of coordinates {coords.shape[0]} != number of rcs {rcs.shape[0]}"
        )

    if point_target_ids is None:
        point_target_ids = [str(p) for p in range(coords.shape[0])]

    if coords.shape[0] != len(point_target_ids):
        raise io_utils.PointTargetDimensionsMismatchError(
            f"number of coordinates {coords.shape[0]} != number of ids {len(point_target_ids)}"
        )

    out = {}
    for index, coord in enumerate(coords):
        out[point_target_ids[index]] = io_utils.NominalPointTarget(
            xyz_coordinates=coord,
            rcs_hh=rcs[index][0],
            rcs_hv=rcs[index][1],
            rcs_vv=rcs[index][2],
            rcs_vh=rcs[index][3],
        )

    return out
