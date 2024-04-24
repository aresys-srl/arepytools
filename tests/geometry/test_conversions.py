# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

from arepytools.geometry.conversions import llh2xyz, xyz2llh
from arepytools.geometry.ellipsoid import WGS84


class ConversionsTestCase(unittest.TestCase):
    xyz = [2.354828227500000e4, 9.457755947560000e05, 6.286558297154000e06]
    llh = [1.422411746702634, 1.545903087718329, 123.4559998866171]

    xyz_vec = [
        [5336078.7743305163, WGS84.semi_major_axis, 0.0, -WGS84.semi_major_axis, 0.0],
        [2346746.5942683504, 0.0, WGS84.semi_major_axis, -WGS84.semi_major_axis, 0.0],
        [4033846.0446414836, 0.0, 0.0, 0.0, WGS84.semi_minor_axis],
    ]

    llh_vec = [
        [0.608159140099359, 0.0, 0.0, 0.0, np.pi / 2],
        [0.414329746479487, 0.0, np.pi / 2, -2.35619449019234, 0.0],
        [717733.676999941, 0.0, 0.0, 2641910.84807364, 0.0],
    ]

    def test_xyz2llh(self):
        llh = xyz2llh(self.xyz)

        self.assertEqual(llh.shape, (3, 1))
        np.testing.assert_allclose(llh.squeeze(), self.llh, atol=1e-8, rtol=1e-10)

    def test_llh2xyz(self):
        xyz = llh2xyz(self.llh)

        self.assertEqual(xyz.shape, (3, 1))
        np.testing.assert_allclose(xyz.squeeze(), self.xyz, atol=1e-8, rtol=1e-10)

    def test_xyz2llh_vectorized(self):
        llh = xyz2llh(self.xyz_vec)

        self.assertEqual(llh.shape, (3, 5))
        np.testing.assert_allclose(llh, self.llh_vec, atol=1e-8, rtol=1e-10)

    def test_llh2xyz_vectorized(self):
        xyz = llh2xyz(self.llh_vec)

        self.assertEqual(xyz.shape, (3, 5))
        np.testing.assert_allclose(xyz, self.xyz_vec, atol=1e-8, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
