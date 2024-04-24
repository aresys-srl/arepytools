# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for io/point_target_binary functionalities"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import arepytools.io.io_support as io_utils
import arepytools.io.point_target_binary as ptb


def _check_product_files(path: Path) -> None:
    """Checking files inside the product folder binary.

    Parameters
    ----------
    path : Path
        path to the product
    """
    assert path.exists()

    rasters = ptb._COORDINATES_RASTER_FILENAMES + ptb._RCS_RASTER_FILENAMES
    rasters = [path.joinpath(r) for r in rasters]
    metadata = [r.with_suffix(".xml") for r in rasters]

    for file in rasters:
        assert file.is_file()
    for file in metadata:
        assert file.is_file()

    lines = []
    for file in metadata:
        raster_info = (
            io_utils.read_metadata(file)
            .get_metadata_channels(0)
            .get_element("RasterInfo")
        )
        lines.append(raster_info.lines)
        assert raster_info.samples == 1

    assert len(set(lines)) == 1


def _check_point_targets(
    points: dict, num: int, coords: np.ndarray, rcs: np.ndarray
) -> None:
    """Checking point targets read from binary product.

    Parameters
    ----------
    points : dict
        point targets from binary product
    num : int
        numerosity
    coords : np.ndarray
        coordinates
    rcs : np.ndarray
        rcs values
    """
    assert isinstance(points, dict)
    assert len(points) == num

    for item in points.values():
        assert isinstance(item, io_utils.NominalPointTarget)
        np.testing.assert_equal(item.xyz_coordinates, coords)
        np.testing.assert_equal(item.rcs_hh, rcs[0])
        np.testing.assert_equal(item.rcs_hv, rcs[1])
        np.testing.assert_equal(item.rcs_vv, rcs[2])
        np.testing.assert_equal(item.rcs_vh, rcs[3])


class PointTargetBinaryTest(unittest.TestCase):
    """Testing point_target_binary functions"""

    def setUp(self):
        """Setting up variables for testing"""
        self.coordinates = np.array(
            [2197913.48269014, 1102055.63813337, 5865641.60621928]
        )
        self.rcs = np.array([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j])

        self.N = 10
        self.M = 4

    def test_writing_point_target_binary_1pt(self) -> None:
        """Testing creation of point target binary product"""
        with TemporaryDirectory() as tmpdir:
            prod_path = Path(tmpdir).joinpath("test")

            prod = ptb.PointSetProduct(prod_path, open_mode="w")
            prod.write_data(coords=self.coordinates, rcs=self.rcs)

            _check_product_files(path=prod_path)

    def test_writing_point_target_binary_npt(self) -> None:
        """Testing creation of point target binary product"""
        with TemporaryDirectory() as tmpdir:
            prod_path = Path(tmpdir).joinpath("test")

            prod = ptb.PointSetProduct(prod_path, open_mode="w")
            prod.write_data(
                coords=np.full((self.N, 3), self.coordinates),
                rcs=np.full((self.N, 4), self.rcs),
            )

            _check_product_files(path=prod_path)

    def test_read_point_target_binary_1pt(self) -> None:
        """Testing creation and read of point target binary product"""
        with TemporaryDirectory() as tmpdir:
            prod_path = Path(tmpdir).joinpath("test")

            prod_0 = ptb.PointSetProduct(prod_path, open_mode="w")
            prod_0.write_data(coords=self.coordinates, rcs=self.rcs)

            _check_product_files(path=prod_path)

            prod_1 = ptb.PointSetProduct(prod_path)
            coords, rcs = prod_1.read_data()

            self.assertEqual(prod_1.number_of_targets, 1)
            np.testing.assert_array_equal(coords, np.atleast_2d(self.coordinates))
            np.testing.assert_array_equal(rcs, np.atleast_2d(self.rcs))

    def test_read_point_target_binary_npt(self) -> None:
        """Testing creation and read of point target binary product"""
        with TemporaryDirectory() as tmpdir:
            prod_path = Path(tmpdir).joinpath("test")

            prod_0 = ptb.PointSetProduct(prod_path, open_mode="w")
            prod_0.write_data(
                coords=np.full((self.N, 3), self.coordinates),
                rcs=np.full((self.N, 4), self.rcs),
            )

            _check_product_files(path=prod_path)

            prod_1 = ptb.PointSetProduct(prod_path)
            coords, rcs = prod_1.read_data()

            self.assertEqual(prod_1.number_of_targets, self.N)
            np.testing.assert_array_equal(
                coords, np.full((self.N, 3), self.coordinates)
            )
            np.testing.assert_array_equal(rcs, np.full((self.N, 4), self.rcs))

    def test_read_point_target_binary_npt_selection(self) -> None:
        """Testing creation and read of point target binary product"""
        with TemporaryDirectory() as tmpdir:
            prod_path = Path(tmpdir).joinpath("test")

            prod_0 = ptb.PointSetProduct(prod_path, open_mode="w")
            prod_0.write_data(
                coords=np.full((self.N, 3), self.coordinates),
                rcs=np.full((self.N, 4), self.rcs),
            )

            _check_product_files(path=prod_path)

            prod_1 = ptb.PointSetProduct(prod_path)
            coords, rcs = prod_1.read_data(start=self.M, num_points=self.M)

            self.assertEqual(prod_1.number_of_targets, self.N)
            np.testing.assert_array_equal(
                coords, np.full((self.M, 3), self.coordinates)
            )
            np.testing.assert_array_equal(rcs, np.full((self.M, 4), self.rcs))

    def test_write_point_target_binary_error1(self) -> None:
        """Testing creation of point target binary product, raising errors"""

        # error: open in read mode, folder does not exist
        with self.assertRaises(io_utils.InvalidPointTargetError):
            with TemporaryDirectory() as tmpdir:
                prod_path = Path(tmpdir).joinpath("prova")
                ptb.PointSetProduct(prod_path)

    def test_write_point_target_binary_error2(self) -> None:
        """Testing creation of point target binary product, raising errors"""

        # error: open in read mode, path not to folder
        with self.assertRaises(io_utils.InvalidPointTargetError):
            with TemporaryDirectory() as tmpdir:
                prod_path = Path(tmpdir).joinpath("prova.xml")
                prod_path.write_text("", encoding="utf-8")
                ptb.PointSetProduct(prod_path)

    def test_write_point_target_binary_error3(self) -> None:
        """Testing creation of point target binary product, raising errors"""

        # error: write wrong shape data
        with self.assertRaises(RuntimeError):
            with TemporaryDirectory() as tmpdir:
                prod_path = Path(tmpdir)

                prod = ptb.PointSetProduct(prod_path, open_mode="w")
                prod.write_data(coords=self.coordinates, rcs=self.rcs.reshape(2, 2))

    def test_write_point_target_binary_error4(self) -> None:
        """Testing creation of point target binary product, raising errors"""

        # error: write wrong shape data
        with self.assertRaises(RuntimeError):
            with TemporaryDirectory() as tmpdir:
                prod_path = Path(tmpdir)

                prod = ptb.PointSetProduct(prod_path, open_mode="w")
                prod.write_data(coords=self.coordinates.reshape(3, 1), rcs=self.rcs)

    def test_write_point_target_binary_error5(self) -> None:
        """Testing creation of point target binary product, raising errors"""

        # error: input number mismatch
        with self.assertRaises(io_utils.PointTargetDimensionsMismatchError):
            with TemporaryDirectory() as tmpdir:
                prod_path = Path(tmpdir)

                prod = ptb.PointSetProduct(prod_path, open_mode="w")
                prod.write_data(
                    coords=np.full((self.N, 3), self.coordinates),
                    rcs=np.full((self.M, 4), self.rcs),
                )


class ConversionToNominalPointTargetTest(unittest.TestCase):
    """Testing convert_array_to_point_target_structure functions"""

    def setUp(self):
        """Setting up variables for testing"""
        self.coordinates = np.array(
            [2197913.48269014, 1102055.63813337, 5865641.60621928]
        )
        self.rcs = np.array([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j])

        self.N = 10
        self.M = 4

    def test_conversion_1pt(self) -> None:
        """Testing conversion to NominalPointTarget dictionary"""
        out = ptb.convert_array_to_point_target_structure(
            coords=self.coordinates, rcs=self.rcs
        )

        _check_point_targets(points=out, coords=self.coordinates, rcs=self.rcs, num=1)

    def test_conversion_1pt_with_id(self) -> None:
        """Testing conversion to NominalPointTarget dictionary"""
        out = ptb.convert_array_to_point_target_structure(
            coords=self.coordinates, rcs=self.rcs, point_target_ids=["5"]
        )

        self.assertEqual(list(out.keys()), ["5"])
        _check_point_targets(points=out, coords=self.coordinates, rcs=self.rcs, num=1)

    def test_conversion_npt(self) -> None:
        """Testing conversion to NominalPointTarget dictionary"""
        out = ptb.convert_array_to_point_target_structure(
            coords=np.full((self.N, 3), self.coordinates),
            rcs=np.full((self.N, 4), self.rcs),
        )

        _check_point_targets(
            points=out, coords=self.coordinates, rcs=self.rcs, num=self.N
        )

    def test_conversion_npt_with_id(self) -> None:
        """Testing conversion to NominalPointTarget dictionary"""
        ids = list(map(chr, range(97, 97 + self.N)))
        out = ptb.convert_array_to_point_target_structure(
            coords=np.full((self.N, 3), self.coordinates),
            rcs=np.full((self.N, 4), self.rcs),
            point_target_ids=ids,
        )

        self.assertEqual(list(out.keys()), ids)
        _check_point_targets(
            points=out, coords=self.coordinates, rcs=self.rcs, num=self.N
        )

    def test_conversion_error1(self) -> None:
        """Testing conversion to NominalPointTarget dictionary, raising errors"""

        # error: shape mismatch
        with self.assertRaises(io_utils.PointTargetDimensionsMismatchError):
            ptb.convert_array_to_point_target_structure(
                coords=np.full((self.N, 3), self.coordinates),
                rcs=np.full((self.M, 4), self.rcs),
            )

    def test_conversion_error2(self) -> None:
        """Testing conversion to NominalPointTarget dictionary, raising errors"""

        # error: wrong coord shape
        with self.assertRaises(RuntimeError):
            ptb.convert_array_to_point_target_structure(
                coords=self.coordinates.reshape(3, 1), rcs=self.rcs
            )

    def test_conversion_error3(self) -> None:
        """Testing conversion to NominalPointTarget dictionary, raising errors"""

        # error: rcs coord shape
        with self.assertRaises(RuntimeError):
            ptb.convert_array_to_point_target_structure(
                coords=self.coordinates, rcs=self.rcs.reshape(2, 2)
            )

    def test_conversion_error4(self) -> None:
        """Testing conversion to NominalPointTarget dictionary, raising errors"""

        # error: rcs coord shape
        with self.assertRaises(io_utils.PointTargetDimensionsMismatchError):
            ptb.convert_array_to_point_target_structure(
                coords=self.coordinates, rcs=self.rcs, point_target_ids=["1", "2"]
            )


if __name__ == "__main__":
    unittest.main()
