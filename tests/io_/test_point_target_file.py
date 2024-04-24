# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for io/point_target_file functionalities"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import arepytools.io.io_support as support
import arepytools.io.point_target_file as ptf


def _check_point_targets(
    points: dict, num: int, coords: np.ndarray, rcs: np.ndarray, delays: float
) -> None:
    """Checking point targets read from file.

    Parameters
    ----------
    points : dict
        point targets from file
    num : int
        numerosity
    coords : np.ndarray
        coordinates
    rcs : np.ndarray
        rcs values
    delays : float
        delay value
    """
    assert isinstance(points, dict)
    assert len(points) == num
    for item in points.values():
        assert isinstance(item, support.NominalPointTarget)
        np.testing.assert_equal(item.xyz_coordinates, coords)
        np.testing.assert_equal(item.rcs_hh, rcs[0])
        np.testing.assert_equal(item.rcs_hv, rcs[1])
        np.testing.assert_equal(item.rcs_vv, rcs[2])
        np.testing.assert_equal(item.rcs_vh, rcs[3])
        np.testing.assert_equal(item.delay, delays)


class PointTargetFileTest(unittest.TestCase):
    """Testing point_target_file functions"""

    def setUp(self):
        """Setting up variables for testing"""
        self.coordinates = np.array(
            [2197913.48269014, 1102055.63813337, 5865641.60621928]
        )
        self.rcs = np.array([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j])
        self.delays = 5
        self.pt_id = 45
        self.default_point_target = support.NominalPointTarget(
            xyz_coordinates=self.coordinates,
            rcs_hh=self.rcs[0],
            rcs_hv=self.rcs[1],
            rcs_vv=self.rcs[2],
            rcs_vh=self.rcs[3],
            delay=self.delays,
        )

        self.N = 10
        self.M = 6

    def test_write_point_target_file_1pt(self) -> None:
        """Testing point target file creation and dump to disk"""

        with TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir).joinpath("test.xml")
            self.assertFalse(xml_path.is_file())

            ptf.write_point_targets_file(
                filename=xml_path,
                point_targets=self.default_point_target,
                target_type=1,
            )

            # checking results
            self.assertTrue(xml_path.is_file())

    def test_write_point_target_file_npt(self) -> None:
        """Testing point target file creation and dump to disk"""

        with TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir).joinpath("test.xml")
            self.assertFalse(xml_path.is_file())

            ptf.write_point_targets_file(
                filename=xml_path,
                point_targets=[self.default_point_target] * self.N,
                target_type=1,
            )

            # checking results
            self.assertTrue(xml_path.is_file())

    def test_write_read_point_target_file_case0a(self) -> None:
        """Testing point target file creation and dump to disk, case 0a"""

        # case 0a: writing 1 target, no target id
        with TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir).joinpath("test.xml")
            self.assertFalse(xml_path.is_file())

            ptf.write_point_targets_file(
                filename=xml_path,
                point_targets=self.default_point_target,
                target_type=1,
            )

            self.assertTrue(xml_path.is_file())

            point_targets = ptf.read_point_targets_file(xml_file=xml_path)

            # checking results
            _check_point_targets(
                points=point_targets,
                num=1,
                coords=self.coordinates,
                rcs=self.rcs,
                delays=self.delays,
            )

    def test_write_read_point_target_file_case0b(self) -> None:
        """Testing point target file creation and dump to disk, case 0b"""

        # case 0b: writing 1 target, with target id
        with TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir).joinpath("test.xml")
            self.assertFalse(xml_path.is_file())

            ptf.write_point_targets_file(
                filename=xml_path,
                point_targets=self.default_point_target,
                target_type=1,
                point_targets_ids=self.pt_id,
            )

            self.assertTrue(xml_path.is_file())

            point_targets = ptf.read_point_targets_file(xml_file=xml_path)

            # checking results
            self.assertEqual(list(point_targets.keys())[0], str(self.pt_id))
            _check_point_targets(
                points=point_targets,
                num=1,
                coords=self.coordinates,
                rcs=self.rcs,
                delays=self.delays,
            )

    def test_write_read_point_target_file_case1a(self) -> None:
        """Testing point target file creation and dump to disk, case 1a"""

        # case 1a: writing N targets, no ids
        with TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir).joinpath("test.xml")
            self.assertFalse(xml_path.is_file())

            ptf.write_point_targets_file(
                filename=xml_path,
                point_targets=[self.default_point_target] * self.N,
                target_type=1,
            )

            self.assertTrue(xml_path.is_file())

            point_targets = ptf.read_point_targets_file(xml_file=xml_path)

            # checking results
            self.assertEqual(
                list(point_targets.keys()), [str(p) for p in range(1, self.N + 1)]
            )
            _check_point_targets(
                points=point_targets,
                num=self.N,
                coords=self.coordinates,
                rcs=self.rcs,
                delays=self.delays,
            )

    def test_write_read_point_target_file_case1b(self) -> None:
        """Testing point target file creation and dump to disk, case 1b"""

        # case 1b: writing N target, with target ids
        with TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir).joinpath("test.xml")
            self.assertFalse(xml_path.is_file())

            indexes = np.arange(self.N) + 5
            ptf.write_point_targets_file(
                filename=xml_path,
                point_targets=[self.default_point_target] * self.N,
                target_type=1,
                point_targets_ids=indexes,
            )

            self.assertTrue(xml_path.is_file())

            point_targets = ptf.read_point_targets_file(xml_file=xml_path)

            # checking results
            self.assertEqual(list(point_targets.keys()), [str(p) for p in indexes])
            _check_point_targets(
                points=point_targets,
                num=self.N,
                coords=self.coordinates,
                rcs=self.rcs,
                delays=self.delays,
            )

    def test_write_read_point_target_file_error0(self) -> None:
        """Testing point target file creation and dump to disk triggering errors"""

        # error: writing N target, with M ids
        with self.assertRaises(support.PointTargetDimensionsMismatchError):
            with TemporaryDirectory() as tmpdir:
                xml_path = Path(tmpdir).joinpath("test.xml")
                self.assertFalse(xml_path.is_file())

                ptf.write_point_targets_file(
                    filename=xml_path,
                    point_targets=[self.default_point_target] * self.N,
                    target_type=1,
                    point_targets_ids=np.arange(self.M),
                )

    def test_write_read_point_target_file_error1(self) -> None:
        """Testing point target file creation and dump to disk triggering errors"""

        # error: writing xml file but already exists
        with self.assertRaises(support.InvalidPointTargetError):
            with TemporaryDirectory() as tmpdir:
                xml_path = Path(tmpdir).joinpath("test.xml")
                xml_path.write_text("", encoding="utf-8")

                ptf.write_point_targets_file(
                    filename=xml_path,
                    point_targets=self.default_point_target,
                    target_type=1,
                )

    def test_write_read_point_target_file_error2(self) -> None:
        """Testing point target file creation and dump to disk triggering errors"""

        # error: writing xml file but filename has no .xml in it
        with self.assertRaises(support.InvalidPointTargetError):
            with TemporaryDirectory() as tmpdir:
                xml_path = Path(tmpdir).joinpath("test")

                ptf.write_point_targets_file(
                    filename=xml_path,
                    point_targets=self.default_point_target,
                    target_type=1,
                )


if __name__ == "__main__":
    unittest.main()
