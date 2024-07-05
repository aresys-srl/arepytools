# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

from arepytools.geometry.rotation import (
    RotationOrder,
    compute_euler_angles_from_rotation,
    compute_rotation,
)


class RotatedFramesTestCase(unittest.TestCase):
    """Testing compute_rotation function"""

    def setUp(self):
        self.yaw = np.deg2rad(10)
        self.pitch = np.deg2rad(15)
        self.roll = np.deg2rad(20)

    def test_compute_rotation(self):
        """Testing compute rotation, scalar inputs"""
        reference_ypr = np.asarray(
            [
                [0.951251242564198, -0.075999422127131, 0.298906609756981],
                [0.167731259496521, 0.940788145499406, -0.294591055321609],
                [-0.258819045102521, 0.330366089549352, 0.907673371190369],
            ],
            dtype=float,
        )
        rotation_matrix_ypr = compute_rotation(
            "YPR",
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
        )
        np.testing.assert_allclose(
            rotation_matrix_ypr.as_matrix(), reference_ypr, rtol=1e-10, atol=1e-10
        )

        reference_yrp = np.asarray(
            [
                [0.935879675463115, -0.163175911166535, 0.312254471657374],
                [0.254907748535925, 0.925416578398323, -0.280403630792980],
                [-0.243210346801694, 0.342020143325669, 0.907673371190369],
            ],
            dtype=float,
        )
        rotation_matrix_yrp = compute_rotation(
            "YRP",
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
        )
        np.testing.assert_allclose(
            rotation_matrix_yrp.as_matrix(), reference_yrp, rtol=1e-10, atol=1e-10
        )

        reference_rpy = np.asarray(
            [
                [0.951251242564198, -0.167731259496521, 0.258819045102521],
                [0.250352400205939, 0.910045011297241, -0.330366089549352],
                [-0.180124260529211, 0.379057122345321, 0.907673371190369],
            ],
            dtype=float,
        )
        rotation_matrix_rpy = compute_rotation(
            "RPY",
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
        )
        np.testing.assert_allclose(
            rotation_matrix_rpy.as_matrix(), reference_rpy, rtol=1e-10, atol=1e-10
        )

        reference_ryp = np.asarray(
            [
                [0.951251242564198, -0.173648177666930, 0.254887002244179],
                [0.246137153725384, 0.925416578398323, -0.288133056037496],
                [-0.185842877388499, 0.336824088833465, 0.923044938291451],
            ],
            dtype=float,
        )
        rotation_matrix_ryp = compute_rotation(
            "RYP",
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
        )
        np.testing.assert_allclose(
            rotation_matrix_ryp.as_matrix(), reference_ryp, rtol=1e-10, atol=1e-10
        )

        reference_pry = np.asarray(
            [
                [0.966622809665280, -0.080554770457117, 0.243210346801694],
                [0.163175911166535, 0.925416578398323, -0.342020143325669],
                [-0.197519532830984, 0.370290541848075, 0.907673371190369],
            ],
            dtype=float,
        )
        rotation_matrix_pry = compute_rotation(
            "PRY",
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
        )
        np.testing.assert_allclose(
            rotation_matrix_pry.as_matrix(), reference_pry, rtol=1e-10, atol=1e-10
        )

        reference_pyr = np.asarray(
            [
                [0.951251242564198, -0.069094499922630, 0.300577816214889],
                [0.173648177666930, 0.925416578398323, -0.336824088833465],
                [-0.254887002244179, 0.372599123061208, 0.892301804089286],
            ],
            dtype=float,
        )
        rotation_matrix_pyr = compute_rotation(
            "PYR",
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
        )
        np.testing.assert_allclose(
            rotation_matrix_pyr.as_matrix(), reference_pyr, rtol=1e-10, atol=1e-10
        )

    def test_compute_rotation_vectorized(self):
        """Testing compute_rotation, array inputs"""
        reference_pyr = np.asarray(
            [
                [0.951251242564198, -0.069094499922630, 0.300577816214889],
                [0.173648177666930, 0.925416578398323, -0.336824088833465],
                [-0.254887002244179, 0.372599123061208, 0.892301804089286],
            ],
            dtype=float,
        )

        rotation_matrix_pyr = compute_rotation(
            "PYR",
            yaw=np.asarray(self.yaw).reshape((1,)),
            pitch=np.asarray(self.pitch).reshape((1,)),
            roll=np.asarray(self.roll).reshape((1,)),
        )

        np.testing.assert_allclose(
            rotation_matrix_pyr.as_matrix(),
            reference_pyr.reshape((1, 3, 3)),
            rtol=1e-10,
            atol=1e-10,
        )

        rotation_matrix_pyr = compute_rotation(
            "PYR",
            yaw=np.asarray([self.yaw, self.yaw]),
            pitch=np.asarray([self.pitch, self.pitch]),
            roll=np.asarray([self.roll, self.roll]),
        )

        np.testing.assert_allclose(
            rotation_matrix_pyr.as_matrix(),
            np.tile(reference_pyr, (2, 1, 1)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_compute_rotation_enum_interface(self):
        """Testing compute_rotation, enum rotation order"""
        rotation_one = compute_rotation(
            "PYR", yaw=self.yaw, pitch=self.pitch, roll=self.roll
        )
        rotation_two = compute_rotation(
            RotationOrder.pyr, yaw=self.yaw, pitch=self.pitch, roll=self.roll
        )
        np.testing.assert_allclose(
            rotation_one.as_matrix(), rotation_two.as_matrix(), rtol=1e-10, atol=1e-10
        )

    def test_compute_rotation_invalid_orders(self):
        """Testing compute_rotation, invalid rotation orders"""
        with self.assertRaises(ValueError):
            compute_rotation("PPP", yaw=0, pitch=0, roll=0)

        with self.assertRaises(ValueError):
            compute_rotation("HYX", yaw=0, pitch=0, roll=0)

        with self.assertRaises(ValueError):
            compute_rotation("unknown order", yaw=0, pitch=0, roll=0)

        with self.assertRaises(ValueError):
            compute_rotation(None, yaw=0, pitch=0, roll=0)

        with self.assertRaises(ValueError):
            compute_rotation("xyz", yaw=0, pitch=0, roll=0)


class EulerAnglesTestCase(unittest.TestCase):
    """Testing compute_euler_angles_from_rotation function"""

    def setUp(self):
        self.yaw = [np.deg2rad(10), np.deg2rad(20), np.deg2rad(30)]
        self.pitch = [np.deg2rad(15), np.deg2rad(25), np.deg2rad(35)]
        self.roll = [np.deg2rad(0), np.deg2rad(30), np.deg2rad(60)]
        self.tolerance = 1e-9

    def test_compute_euler_angles_scalar(self):
        """Testing compute_euler_angles for single values of yaw, pitch and roll"""
        rotation = compute_rotation(
            "YPR", yaw=self.yaw[0], pitch=self.pitch[0], roll=self.roll[0]
        )
        euler_angles = compute_euler_angles_from_rotation("YPR", rotation=rotation)

        np.testing.assert_allclose(
            self.yaw[0], euler_angles[0], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            self.pitch[0], euler_angles[1], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            self.roll[0], euler_angles[2], atol=self.tolerance, rtol=0
        )

    def test_compute_euler_angles_scalar_enum(self):
        """Testing compute_euler_angles for single values of yaw, pitch and roll"""
        rotation = compute_rotation(
            RotationOrder.ypr, yaw=self.yaw[0], pitch=self.pitch[0], roll=self.roll[0]
        )
        euler_angles = compute_euler_angles_from_rotation(
            RotationOrder.ypr, rotation=rotation
        )

        np.testing.assert_allclose(
            self.yaw[0], euler_angles[0], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            self.pitch[0], euler_angles[1], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            self.roll[0], euler_angles[2], atol=self.tolerance, rtol=0
        )

    def test_compute_euler_angles_vectorized(self):
        """Testing compute_euler_angles for arrays of yaw, pitch and roll"""
        rotation = compute_rotation(
            "YPR", yaw=self.yaw, pitch=self.pitch, roll=self.roll
        )
        euler_angles = compute_euler_angles_from_rotation("YPR", rotation=rotation)

        np.testing.assert_allclose(
            self.yaw, euler_angles[:, 0], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            self.pitch, euler_angles[:, 1], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            self.roll, euler_angles[:, 2], atol=self.tolerance, rtol=0
        )

    def test_compute_euler_angles_vectorized_all_rotations(self):
        """Testing compute_euler_angles for arrays of yaw, pitch and roll, all rotation orders"""
        for order in RotationOrder:
            rotation = compute_rotation(
                order, yaw=self.yaw, pitch=self.pitch, roll=self.roll
            )
            euler_angles = compute_euler_angles_from_rotation(order, rotation=rotation)
            np.testing.assert_allclose(
                self.yaw,
                euler_angles[:, order.value.find("Y")],
                atol=self.tolerance,
                rtol=0,
            )
            np.testing.assert_allclose(
                self.pitch,
                euler_angles[:, order.value.find("P")],
                atol=self.tolerance,
                rtol=0,
            )
            np.testing.assert_allclose(
                self.roll,
                euler_angles[:, order.value.find("R")],
                atol=self.tolerance,
                rtol=0,
            )

    def test_compute_euler_angles_invalid_orders(self):
        """Testing compute_euler_angles with invalid rotation orders"""
        rotation = compute_rotation(
            "YPR", yaw=self.yaw, pitch=self.pitch, roll=self.roll
        )
        with self.assertRaises(ValueError):
            compute_euler_angles_from_rotation("PPP", rotation=rotation)

        with self.assertRaises(ValueError):
            compute_euler_angles_from_rotation("HYX", rotation=rotation)

        with self.assertRaises(ValueError):
            compute_euler_angles_from_rotation("unknown order", rotation=rotation)

        with self.assertRaises(ValueError):
            compute_euler_angles_from_rotation(None, rotation=rotation)

        with self.assertRaises(ValueError):
            compute_euler_angles_from_rotation("xyz", rotation=rotation)


if __name__ == "__main__":
    unittest.main()
