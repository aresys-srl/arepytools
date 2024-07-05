# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing arepytools/geometry/attitude_utils module"""

import unittest

import numpy as np

from arepytools.geometry.attitude_utils import (
    compute_antenna_reference_frame_from_euler_angles,
    compute_euler_angles_from_antenna_reference_frame,
)

initial_frame_axis = np.array(
    [
        [
            [-6.96355200e-01, -1.71146342e-01, -6.96992371e-01],
            [-1.20963385e-01, 9.85245619e-01, -1.21074068e-01],
            [7.07430064e-01, -8.82912472e-13, -7.06783351e-01],
        ],
        [
            [-6.96464643e-01, -1.71146342e-01, -6.96883010e-01],
            [-1.20982397e-01, 9.85245619e-01, -1.21055071e-01],
            [7.07319065e-01, -9.79697300e-14, -7.06894433e-01],
        ],
        [
            [-6.96574070e-01, -1.71146342e-01, -6.96773632e-01],
            [-1.21001405e-01, 9.85245619e-01, -1.21036071e-01],
            [7.07208049e-01, 1.89524695e-13, -7.07005499e-01],
        ],
        [
            [-6.96683480e-01, -1.71146342e-01, -6.96664237e-01],
            [-1.21020410e-01, 9.85245619e-01, -1.21017068e-01],
            [7.07097016e-01, 5.97549244e-14, -7.07116547e-01],
        ],
        [
            [-6.96792872e-01, -1.71146342e-01, -6.96554824e-01],
            [-1.21039413e-01, 9.85245619e-01, -1.20998062e-01],
            [7.06985964e-01, 9.86256525e-14, -7.07227578e-01],
        ],
        [
            [-6.96902248e-01, -1.71146342e-01, -6.96445394e-01],
            [-1.21058413e-01, 9.85245619e-01, -1.20979053e-01],
            [7.06874895e-01, 5.04007487e-13, -7.07338591e-01],
        ],
        [
            [-6.97011607e-01, -1.71146342e-01, -6.96335946e-01],
            [-1.21077409e-01, 9.85245619e-01, -1.20960041e-01],
            [7.06763808e-01, -7.40657615e-13, -7.07449588e-01],
        ],
        [
            [-6.97120949e-01, -1.71146342e-01, -6.96226481e-01],
            [-1.21096403e-01, 9.85245619e-01, -1.20941026e-01],
            [7.06652704e-01, -6.89212391e-14, -7.07560567e-01],
        ],
        [
            [-6.97230274e-01, -1.71146342e-01, -6.96116998e-01],
            [-1.21115394e-01, 9.85245619e-01, -1.20922007e-01],
            [7.06541582e-01, 6.75836064e-13, -7.07671529e-01],
        ],
        [
            [-6.97339581e-01, -1.71146342e-01, -6.96007498e-01],
            [-1.21134381e-01, 9.85245619e-01, -1.20902986e-01],
            [7.06430442e-01, -7.27116637e-14, -7.07782474e-01],
        ],
    ]
)
yaw = np.repeat(-0.1908572110448003, 10)
pitch = np.repeat(-0.34811037898980685, 10)
roll = np.repeat(-0.9970240464237335, 10)
arf = np.array(
    [
        [
            [-0.91097982, 0.32557499, -0.25321275],
            [0.02274693, 0.65264641, 0.7573211],
            [0.4118232, 0.68414442, -0.60195354],
        ],
        [
            [-0.91104352, 0.32546914, -0.25311961],
            [0.02273586, 0.65262802, 0.75733727],
            [0.41168286, 0.68421233, -0.60197235],
        ],
        [
            [-0.91110721, 0.32536327, -0.25302648],
            [0.0227248, 0.65260963, 0.75735345],
            [0.41154251, 0.68428021, -0.60199115],
        ],
        [
            [-0.91117087, 0.3252574, -0.25293334],
            [0.02271374, 0.65259124, 0.75736963],
            [0.41140215, 0.68434809, -0.60200994],
        ],
        [
            [-0.91123451, 0.32515151, -0.25284019],
            [0.02270269, 0.65257285, 0.75738581],
            [0.41126178, 0.68441594, -0.60202871],
        ],
        [
            [-0.91129813, 0.32504561, -0.25274705],
            [0.02269163, 0.65255445, 0.75740199],
            [0.4111214, 0.68448378, -0.60204747],
        ],
        [
            [-0.91136173, 0.3249397, -0.2526539],
            [0.02268059, 0.65253605, 0.75741817],
            [0.41098101, 0.6845516, -0.60206621],
        ],
        [
            [-0.9114253, 0.32483378, -0.25256074],
            [0.02266954, 0.65251765, 0.75743436],
            [0.41084061, 0.6846194, -0.60208493],
        ],
        [
            [-0.91148886, 0.32472785, -0.25246759],
            [0.0226585, 0.65249925, 0.75745054],
            [0.4107002, 0.68468719, -0.60210365],
        ],
        [
            [-0.91155239, 0.32462191, -0.25237443],
            [0.02264747, 0.65248085, 0.75746672],
            [0.41055978, 0.68475496, -0.60212234],
        ],
    ]
)
tolerance = 1e-8


class ARFFromYPRTest(unittest.TestCase):
    """Testing compute_antenna_reference_frame_from_euler_angles function"""

    def test_compute_arf_from_euler_angles(self):
        """Testing ARF computation from euler angles and initial frame axis"""
        computed_arf = compute_antenna_reference_frame_from_euler_angles(
            order="RPY",
            initial_reference_frame_axis=initial_frame_axis,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
        np.testing.assert_allclose(computed_arf, arf, atol=tolerance, rtol=0)


class YPRFromARFTest(unittest.TestCase):
    """Testing compute_euler_angles_from_antenna_reference_frame function"""

    def setUp(self): ...

    def test_compute_euler_angles_from_arf_single_value(self):
        """Testing euler angles computation from arf and initial frame axis"""
        euler_angles = compute_euler_angles_from_antenna_reference_frame(
            initial_reference_frame_axis=initial_frame_axis[0, :, :],
            antenna_reference_frame=arf[0, :, :],
            order="RPY",
        )
        np.testing.assert_allclose(euler_angles[0], roll[0], atol=tolerance, rtol=0)
        np.testing.assert_allclose(euler_angles[1], pitch[0], atol=tolerance, rtol=0)
        np.testing.assert_allclose(euler_angles[2], yaw[0], atol=tolerance, rtol=0)

    def test_compute_euler_angles_from_arf(self):
        """Testing euler angles computation from arf and initial frame axis"""
        euler_angles = compute_euler_angles_from_antenna_reference_frame(
            initial_reference_frame_axis=initial_frame_axis,
            antenna_reference_frame=arf,
            order="RPY",
        )
        np.testing.assert_allclose(euler_angles[:, 0], roll, atol=tolerance, rtol=0)
        np.testing.assert_allclose(euler_angles[:, 1], pitch, atol=tolerance, rtol=0)
        np.testing.assert_allclose(euler_angles[:, 2], yaw, atol=tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
