# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/anx_time.py core functionalities"""

import unittest

import numpy as np

from arepytools.geometry.anx_time import (
    _find_anx_time_intervals,
    compute_anx_times_core,
    compute_relative_times,
)
from arepytools.geometry.conversions import llh2xyz
from arepytools.io.create_orbit import create_orbit
from arepytools.io.metadata import StateVectors
from arepytools.timing.precisedatetime import PreciseDateTime

state_vectors = np.array(
    [
        -779001.28129911,
        -505870.7546934,
        6817987.21562447,
        -1787813.04699317,
        629706.49654234,
        6614777.74861071,
        -2676777.49486916,
        1762403.33562427,
        6089174.4622466,
        -3403702.99091361,
        2833140.93065613,
        5266805.03570274,
        -3935963.30795762,
        3784762.95672522,
        4187763.77972863,
        -4251980.19525865,
        4565112.00564628,
        2904652.08879896,
        -4342081.19344017,
        5129895.41407931,
        1480008.94949346,
        -4208699.84360167,
        5445182.53365344,
        -16741.96566374,
        -3865922.04471967,
        5489393.41476849,
        -1512678.67406476,
        -3338417.95554259,
        5254664.63265023,
        -2934934.90842229,
        -2659831.54067513,
        4747510.15094551,
        -4214248.33821238,
        -1870727.8761017,
        3988731.95503892,
        -5288330.41117336,
        -1016220.22596924,
        3012574.72049191,
        -6104894.47001853,
        -143413.63392233,
        1865158.91592616,
        -6624195.83061395,
        701191.28183798,
        602265.36994385,
        -6820961.43577,
        1474191.68582983,
        -713420.57171269,
        -6685616.32885961,
        2137539.09251902,
        -2015467.38227689,
        -6224748.10095948,
        2660441.17922355,
        -3237097.45252941,
        -5460787.06866317,
        3020812.70959749,
        -4314653.48632447,
        -4430917.553466,
        3206206.3232972,
        -5190950.00548857,
        -3185272.53864321,
        3214184.02923558,
        -5818332.44350126,
        -1784498.48374172,
        3052122.01979767,
        -6161279.32465417,
        -296807.57728266,
        2736473.1183581,
        -6198404.88017072,
        1205340.24567249,
        2291541.0597032,
        -5923749.16632974,
        2648764.83733911,
        1747847.25026194,
        -5347278.90699311,
        3963131.52602845,
        1140192.21257602,
        -4494563.084553,
        5084381.11411162,
        505529.40108162,
        -3405630.64317499,
        5957856.38589271,
        -119222.36344083,
        -2133061.23760086,
        6540971.68296547,
        -699300.40803076,
        -739401.37358005,
        6805294.15498011,
    ]
)
origin = PreciseDateTime.from_utc_string("27-JUL-2021 23:28:48.000005006790")


height1 = 800000
state_vectors1_pos = np.array(
    [
        llh2xyz([-1e-4, 0, height1]),
        llh2xyz([5e-5, 1e-4, height1]),
        llh2xyz([2e-4, 2e-4, height1]),
        llh2xyz([1e-4, 3e-4, height1]),
        llh2xyz([0, 4e-4, height1]),
        llh2xyz([-1e-4, 5e-4, height1]),
        llh2xyz([0, 6e-4, height1]),
        llh2xyz([1e-4, 7e-4, height1]),
        llh2xyz([5e-5, 8e-4, height1]),
        llh2xyz([-5e-5, 9e-4, height1]),
    ]
).ravel()
time_axis1 = np.array(
    [
        PreciseDateTime(state_vector_index)
        for state_vector_index in range(state_vectors1_pos.size // 3)
    ]
)
state_vectors1_vel = np.array(
    [
        4.19207049e-01,
        7.17813768e02,
        -4.16233937e02,
        -2.51653896e-01,
        7.17813660e02,
        1.66493583e03,
        -1.64374437e-01,
        7.17813679e02,
        1.78385983e02,
        -1.17975094e-01,
        7.17813688e02,
        -8.62198913e02,
        -2.99017869e-01,
        7.17813635e02,
        -8.32467921e02,
        -3.58906835e-01,
        7.17813611e02,
        1.19371180e-12,
        -4.29944917e-01,
        7.17813567e02,
        9.21660912e02,
        -5.10645570e-01,
        7.17813519e02,
        2.08116980e02,
        -5.27424590e-01,
        7.17813501e02,
        -8.02736923e02,
        -7.44887782e-01,
        7.17813342e02,
        -2.08116983e02,
    ]
)


class ANXTimeFunctionsTest(unittest.TestCase):
    """Testing ANX Time module functionalities"""

    def setUp(self):
        """Setting up variables for testing"""
        self._anx_times = np.array(
            [
                PreciseDateTime.from_utc_string("01-JAN-1985 00:00:00.786117076873"),
                PreciseDateTime.from_utc_string("01-JAN-1985 00:00:06.000000953674"),
            ]
        )
        self._time_points = PreciseDateTime.from_utc_string(
            "01-JAN-1985 00:00:00.000000000000"
        ) + np.arange(10)
        _state_vectors = StateVectors(
            position_vector=state_vectors,
            velocity_vector=np.diff(np.append(state_vectors, state_vectors[-1])),
            t_ref_utc=origin,
            dt_sv_s=200,
        )
        _state_vectors1 = StateVectors(
            position_vector=state_vectors1_pos,
            velocity_vector=state_vectors1_vel,
            t_ref_utc=time_axis1[0],
            dt_sv_s=1,
        )
        self._trajectory = create_orbit(_state_vectors)
        self._trajectory1 = create_orbit(_state_vectors1)

        # expected_results
        self._tolerance = 1e-8
        self._relative_times = np.array(
            [
                np.nan,
                0.21388292,
                1.21388292,
                2.21388292,
                3.21388292,
                4.21388292,
                5.21388292,
                0.99999905,
                1.99999905,
                2.99999905,
            ]
        )
        self._relative_indices = np.array([2, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        self._anx_times_result = PreciseDateTime.from_utc_string(
            "28-JUL-2021 00:39:27.361122507601"
        )
        self._anx_indexes = [(0, 1), (6, 7)]
        self._anx_times_result1 = np.array(
            [
                PreciseDateTime.from_utc_string("01-JAN-1985 00:00:00.790886402130"),
                PreciseDateTime.from_utc_string("01-JAN-1985 00:00:06.000000000000"),
            ]
        )

    def test_compute_relative_times(self) -> None:
        """Testing compute_relative_times function"""
        rel_times, indexes = compute_relative_times(
            time_points=self._time_points, anx_times=self._anx_times
        )
        np.testing.assert_allclose(
            rel_times, self._relative_times, atol=self._tolerance, rtol=0
        )
        np.testing.assert_allclose(
            indexes, self._relative_indices, atol=self._tolerance, rtol=0
        )

    def test_compute_anx_times_core_case0(self) -> None:
        """Testing ANX time computation"""
        anx_times = compute_anx_times_core(
            trajectory=self._trajectory, time_sampling_step_s=200
        )
        time_delta = anx_times - self._anx_times_result
        np.testing.assert_almost_equal(
            time_delta.astype(float), np.zeros_like(anx_times), decimal=8
        )

    def test_compute_anx_times_core_case1(self) -> None:
        """Testing ANX time computation"""
        pos = self._trajectory1.evaluate(time_axis1)
        anx_times_rel = _find_anx_time_intervals(
            time_axis_rel=time_axis1 - time_axis1[0], positions=pos
        )
        anx_times = compute_anx_times_core(
            trajectory=self._trajectory1, time_sampling_step_s=1
        )
        time_delta = anx_times - self._anx_times_result1

        self.assertEqual(len(anx_times_rel), 2)
        self.assertListEqual(anx_times_rel, self._anx_indexes)
        self.assertEqual(anx_times.size, 2)
        np.testing.assert_almost_equal(
            time_delta.astype(float), np.zeros_like(anx_times), decimal=8
        )


if __name__ == "__main__":
    unittest.main()
