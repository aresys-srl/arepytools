# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing arepytools/geometry/geometric_functions.py functionalities"""

import itertools
import unittest
from unittest.case import TestCase

import numpy as np

from arepytools.geometry.generalsarorbit import GeneralSarOrbit, GSO3DCurveWrapper
from arepytools.geometry.geometric_functions import (
    compute_incidence_angles,
    compute_incidence_angles_from_trajectory,
    compute_look_angles,
    compute_look_angles_from_trajectory,
    get_geometric_squint,
)
from arepytools.math.axis import RegularAxis
from arepytools.timing.precisedatetime import PreciseDateTime


class ComputeLookAndIncidenceAnglesTest(TestCase):
    """Testing compute_incidence_angles and compute_look_angles functions"""

    sensor_positions = np.array(
        [
            [5317606.94350283, 610603.985945038, 4577936.89859885],
            [5313024.53547427, 608285.563877273, 4583547.15708167],
            [5308435.7651548, 605967.120830312, 4589152.18047604],
            [5303840.63790599, 603648.660435838, 4594751.96221552],
            [5299239.15894225, 601330.18624638, 4600346.49592944],
            [5294631.33350784, 599011.701824865, 4605935.7752263],
            [5290017.16682646, 596693.210719223, 4611519.79375494],
        ]
    )

    points = 1.0e6 * np.array(
        [
            [4.759710115562946, 0.723739860905043, 4.169511582485821],
            [4.740767822131609, 0.785940591895703, 4.179749794811581],
            [4.719765088454115, 0.852956178108469, 4.190295797874668],
            [4.695901593234693, 0.926811645389169, 4.201333096521494],
            [4.668001681937613, 1.010339507423213, 4.213072517214957],
            [4.634228769201612, 1.107766560515324, 4.225761582103892],
            [4.591484972384566, 1.225899798385040, 4.239684548934286],
        ]
    )

    surface_normals = np.array(
        [
            [0.234003747210404, 0.035581544956606, 0.206369066952520],
            [0.233072478806222, 0.038639547183139, 0.206875805039975],
            [0.232039912061006, 0.041934264280303, 0.207397777162885],
            [0.230866700422629, 0.045565253495203, 0.207944065853625],
            [0.229495044663803, 0.049671770959056, 0.208525105922023],
            [0.227834651920851, 0.054461620540163, 0.209153148422883],
            [0.225733219610407, 0.060269457500911, 0.209842262631383],
        ]
    )

    nadir_directions = np.array(
        [
            [-5.076811128664492, -0.582954164154652, -4.397261878693197],
            [-5.072502925338587, -0.580748363122939, -4.402708576084343],
            [-5.068188606556682, -0.578542492291151, -4.408150450773756],
            [-5.063868176538823, -0.576336554975789, -4.413587496083914],
            [-5.059541639367873, -0.574130554419002, -4.419019705524370],
            [-5.055208999156272, -0.571924493873826, -4.424447072581663],
            [-5.050870259998338, -0.569718376579988, -4.429869590778411],
        ]
    )

    def test_compute_incidence_angles_1_pos_1_point(self):
        """Testing compute incidence angles with scalar inputs"""
        reference_value = np.array([0.289504602345834])

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]

        point_inputs = [
            self.points[0],
            self.points[0].reshape((1, 3)),
        ]

        for position, point in itertools.product(position_inputs, point_inputs):
            incidence_angle = compute_incidence_angles(position, point)
            np.testing.assert_allclose(
                reference_value, incidence_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_incidence_angles_1_pos_1_point_1_surface_normal(self):
        """Testing compute incidence angles with scalar inputs"""
        reference_value = np.array([0.290207579201514])

        surface_normals = [
            (self.surface_normals[0], None),
            (self.surface_normals[0], False),
            (self.surface_normals[0].reshape((1, 3)), None),
            (self.surface_normals[0].reshape((1, 3)), False),
            (self.surface_normals[0] / np.linalg.norm(self.surface_normals[0]), True),
        ]

        for surface_normal, assume_normalized in surface_normals:
            options = (
                {}
                if assume_normalized is None
                else {"assume_surface_normals_normalized": assume_normalized}
            )

            incidence_angle = compute_incidence_angles(
                self.sensor_positions[0],
                self.points[0],
                surface_normals=surface_normal,
                **options,
            )

            np.testing.assert_allclose(
                reference_value, incidence_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_incidence_angles_1_pos_N_point(self):
        """Testing compute incidence angles with mixed inputs"""
        reference_value = np.array(
            [
                0.289504602345834,
                0.387105058129672,
                0.485521632720666,
                0.585097084845005,
                0.686313174870685,
                0.789892511505193,
                0.897005905237912,
            ]
        )

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]

        for position in position_inputs:
            incidence_angle = compute_incidence_angles(position, self.points)
            np.testing.assert_allclose(
                reference_value, incidence_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_incidence_angles_1_pos_N_point_N_surface_normal(self):
        """Testing compute incidence angles with mixed inputs"""
        reference_value = np.array(
            [
                0.290207579201514,
                0.387779384509382,
                0.486164927958404,
                0.585705866765378,
                0.686882472903164,
                0.790415131872372,
                0.897471035962306,
            ]
        )

        surface_normals = [
            (self.surface_normals, None),
            (self.surface_normals, False),
            (
                self.surface_normals
                / np.linalg.norm(self.surface_normals, axis=-1, keepdims=True),
                True,
            ),
        ]

        for surface_normal, assume_normalized in surface_normals:
            options = (
                {}
                if assume_normalized is None
                else {"assume_surface_normals_normalized": assume_normalized}
            )
            incidence_angle = compute_incidence_angles(
                self.sensor_positions[0],
                self.points,
                surface_normals=surface_normal,
                **options,
            )

            np.testing.assert_allclose(
                reference_value, incidence_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_incidence_angles_N_pos_N_point(self):
        """Testing compute incidence angles with array inputs"""
        reference_value = np.array(
            [
                0.289504602345834,
                0.387283126037465,
                0.485979187965376,
                0.585817587250902,
                0.687224085709431,
                0.790895812723984,
                0.897997938422811,
            ]
        )

        incidence_angle = compute_incidence_angles(self.sensor_positions, self.points)
        np.testing.assert_allclose(
            reference_value, incidence_angle, rtol=1e-10, atol=1e-10
        )

    def test_compute_incidence_angles_N_pos_N_point_N_surface_normal(
        self,
    ):
        """Testing compute incidence angles with array inputs"""
        reference_value = np.array(
            [
                0.290207579201514,
                0.387866437194178,
                0.486480609230522,
                0.586255132207420,
                0.687606324073378,
                0.791225249434776,
                0.898271544290844,
            ]
        )

        surface_normals = [
            (self.surface_normals, None),
            (self.surface_normals, False),
            (
                self.surface_normals
                / np.linalg.norm(self.surface_normals, axis=-1, keepdims=True),
                True,
            ),
        ]

        for surface_normal, assume_normalized in surface_normals:
            options = (
                {}
                if assume_normalized is None
                else {"assume_surface_normals_normalized": assume_normalized}
            )

            incidence_angle = compute_incidence_angles(
                self.sensor_positions,
                self.points,
                surface_normals=surface_normal,
                **options,
            )

            np.testing.assert_allclose(
                reference_value, incidence_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_incidence_angles_invalid_inputs(self):
        """Testing compute incidence angles with invalid inputs"""
        # wrong sensor positions shape
        with self.assertRaises(ValueError):
            compute_incidence_angles(np.arange(5), np.arange(3))
        with self.assertRaises(ValueError):
            compute_incidence_angles(np.arange(6).reshape(3, 2), np.arange(3))

        # wrong points shape
        with self.assertRaises(ValueError):
            compute_incidence_angles(np.arange(3), np.arange(5))
        with self.assertRaises(ValueError):
            compute_incidence_angles(np.arange(3), np.arange(6).reshape(3, 2))

        # incompatible points sensor positions shape
        with self.assertRaises(ValueError):
            compute_incidence_angles(
                np.arange(12).reshape(4, 3), np.arange(6).reshape(2, 3)
            )

        # incompatible points surface normals shape
        with self.assertRaises(ValueError):
            compute_incidence_angles(
                np.arange(3),
                np.arange(12).reshape(4, 3),
                surface_normals=np.arange(12).reshape((6, 2)),
            )

    def test_compute_look_angles_1_pos_1_nadir_1_point(self):
        """Testing compute look angles with scalar inputs"""
        reference_value = np.array([0.261807718170898])

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]
        nadir_dir_inputs = [
            (self.nadir_directions[0], None),
            (self.nadir_directions[0].reshape((1, 3)), None),
            (self.nadir_directions[0], False),
            (self.nadir_directions[0].reshape((1, 3)), False),
            (self.nadir_directions[0] / np.linalg.norm(self.nadir_directions[0]), True),
        ]
        points_inputs = [
            self.points[0],
            self.points[0].reshape((1, 3)),
        ]

        for position, (nadir_dir, assume_normalized), point in itertools.product(
            position_inputs, nadir_dir_inputs, points_inputs
        ):
            options = (
                {}
                if assume_normalized is None
                else {"assume_nadir_directions_normalized": assume_normalized}
            )
            look_angle = compute_look_angles(position, nadir_dir, point, **options)

            np.testing.assert_allclose(
                reference_value, look_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_look_angles_1_pos_1_nadir_N_point(self):
        """Testing compute look angles with mixed inputs"""
        reference_value = np.array(
            [
                0.261807718170898,
                0.349072925268265,
                0.436338618964629,
                0.523604555599314,
                0.610870630659514,
                0.698136791805967,
                0.785403009878366,
            ]
        )

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]
        nadir_dir_inputs = [
            (self.nadir_directions[0], None),
            (self.nadir_directions[0].reshape((1, 3)), None),
            (self.nadir_directions[0], False),
            (self.nadir_directions[0].reshape((1, 3)), False),
            (self.nadir_directions[0] / np.linalg.norm(self.nadir_directions[0]), True),
        ]

        for position, (nadir_dir, assume_normalized) in itertools.product(
            position_inputs, nadir_dir_inputs
        ):
            options = (
                {}
                if assume_normalized is None
                else {"assume_nadir_directions_normalized": assume_normalized}
            )

            look_angle = compute_look_angles(
                position, nadir_dir, self.points, **options
            )

            np.testing.assert_allclose(
                reference_value, look_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_look_angles_N_pos_1_nadir_N_point(self):
        """Testing compute look angles with mixed inputs"""
        reference_value = np.array(
            [
                0.261807718170898,
                0.349177447165984,
                0.436707504824461,
                0.524249216838951,
                0.611732608815392,
                0.699124457083078,
                0.786414020130282,
            ]
        )

        nadir_dir_inputs = [
            (self.nadir_directions[0], None),
            (self.nadir_directions[0].reshape((1, 3)), None),
            (self.nadir_directions[0], False),
            (self.nadir_directions[0].reshape((1, 3)), False),
            (self.nadir_directions[0] / np.linalg.norm(self.nadir_directions[0]), True),
        ]

        for nadir_dir, assume_normalized in nadir_dir_inputs:
            options = (
                {}
                if assume_normalized is None
                else {"assume_nadir_directions_normalized": assume_normalized}
            )

            look_angle = compute_look_angles(
                self.sensor_positions, nadir_dir, self.points, **options
            )

            np.testing.assert_allclose(
                reference_value, look_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_look_angles_1_pos_N_nadir_N_point(self):
        """Testing compute look angles with mixed inputs"""
        reference_value = np.array(
            [
                0.261807718170898,
                0.349080180131945,
                0.436352948981580,
                0.523625656261822,
                0.610898076081333,
                0.698170031320458,
                0.785441361124307,
            ]
        )

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]
        nadir_dir_inputs = [
            (self.nadir_directions, None),
            (self.nadir_directions, False),
            (
                self.nadir_directions
                / np.linalg.norm(self.nadir_directions, axis=-1, keepdims=True),
                True,
            ),
        ]

        for position, (nadir_dir, assume_normalized) in itertools.product(
            position_inputs, nadir_dir_inputs
        ):
            options = (
                {}
                if assume_normalized is None
                else {"assume_nadir_directions_normalized": assume_normalized}
            )

            look_angle = compute_look_angles(
                position, nadir_dir, self.points, **options
            )

            np.testing.assert_allclose(
                reference_value, look_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_look_angles_N_pos_N_nadir_N_point(self):
        """Testing compute look angles with array inputs"""
        reference_value = np.array(
            [
                0.261807718170898,
                0.349151450027951,
                0.436618549676532,
                0.524083903169947,
                0.611489294069069,
                0.698809101292470,
                0.786038760035532,
            ]
        )

        nadir_dir_inputs = [
            (self.nadir_directions, None),
            (self.nadir_directions, False),
            (
                self.nadir_directions
                / np.linalg.norm(self.nadir_directions, axis=-1, keepdims=True),
                True,
            ),
        ]

        for nadir_dir, assume_normalized in nadir_dir_inputs:
            options = (
                {}
                if assume_normalized is None
                else {"assume_nadir_directions_normalized": assume_normalized}
            )

            look_angle = compute_look_angles(
                self.sensor_positions, nadir_dir, self.points, **options
            )

            np.testing.assert_allclose(
                reference_value, look_angle, rtol=1e-10, atol=1e-10
            )

    def test_compute_look_angles_invalid_inputs(self):
        """Testing compute look angles with invalid inputs"""
        # wrong point shape
        with self.assertRaises(ValueError):
            compute_look_angles(
                np.arange(12).reshape(4, 3),
                np.arange(12).reshape(4, 3),
                np.arange(10).reshape(2, 5),
            )

        # wrong position shape
        with self.assertRaises(ValueError):
            compute_look_angles(np.arange(5), np.arange(3), np.arange(3))
        with self.assertRaises(ValueError):
            compute_look_angles(np.arange(6).reshape(3, 2), np.arange(3), np.arange(3))

        # wrong nadir direction shape
        with self.assertRaises(ValueError):
            compute_look_angles(np.arange(3), np.arange(5), np.arange(3))
        with self.assertRaises(ValueError):
            compute_look_angles(np.arange(3), np.arange(6).reshape(3, 2), np.arange(3))

        # incompatible shapes
        with self.assertRaises(ValueError):
            compute_look_angles(
                np.arange(12).reshape(4, 3),
                np.arange(12).reshape(4, 3),
                np.arange(6).reshape(2, 3),
            )
        with self.assertRaises(ValueError):
            compute_look_angles(
                np.arange(12).reshape(4, 3), np.arange(6), np.arange(12).reshape(4, 3)
            )
        with self.assertRaises(ValueError):
            compute_look_angles(
                np.arange(1, 4),
                np.arange(15).reshape((5, 3)),
                np.arange(12).reshape(4, 3),
            )


class ComputeLookIncidenceAnglesFromTrajectoryTest(TestCase):
    """Testing compute_incidence_angles_from_trajectory and compute_look_angles_from_trajectory functions"""

    def setUp(self):
        state_vectors = np.array(
            [
                4454010.19620684,
                773703.063195028,
                4486349.18475909,
                4453313.69416755,
                773582.074302417,
                4487056.65289135,
                4452617.07996439,
                773461.065925896,
                4487764.01275803,
                4451920.35364779,
                773340.038074226,
                4488471.2643057,
                4451223.5152682,
                773218.990756171,
                4489178.40748093,
                4450526.5648761,
                773097.923980499,
                4489885.44223035,
                4449829.50252199,
                772976.837755983,
                4490592.36850059,
                4449132.32825639,
                772855.732091401,
                4491299.18623831,
                4448435.04212988,
                772734.606995533,
                4492005.89539019,
                4447737.64419302,
                772613.462477168,
                4492712.49590294,
                4447040.13449642,
                772492.298545094,
                4493418.98772329,
                4446342.5130907,
                772371.115208107,
                4494125.37079799,
                4445644.78002653,
                772249.912475006,
                4494831.64507384,
                4444946.93535458,
                772128.690354593,
                4495537.81049761,
                4444248.97912556,
                772007.448855678,
                4496243.86701616,
                4443550.91139019,
                771886.187987072,
                4496949.81457631,
                4442852.73219923,
                771764.907757591,
                4497655.65312496,
                4442154.44160346,
                771643.608176057,
                4498361.38260899,
                4441456.03965368,
                771522.289251295,
                4499067.00297534,
                4440757.52640073,
                771400.950992134,
                4499772.51417094,
                4440058.90189545,
                771279.593407408,
                4500477.91614277,
                4439360.16618873,
                771158.216505955,
                4501183.20883781,
                4438661.31933146,
                771036.820296618,
                4501888.3922031,
                4437962.36137459,
                770915.404788244,
                4502593.46618568,
                4437263.29236905,
                770793.969989682,
                4503298.4307326,
                4436564.11236584,
                770672.51590979,
                4504003.28579096,
                4435864.82141595,
                770551.042557426,
                4504708.03130787,
                4435165.41957041,
                770429.549941455,
                4505412.66723048,
                4434465.90688028,
                770308.038070746,
                4506117.19350595,
                4433766.28339662,
                770186.506954169,
                4506821.61008147,
                4433066.54917055,
                770064.956600603,
                4507525.91690424,
                4432366.70425318,
                769943.387018929,
                4508230.1139215,
                4431666.74869568,
                769821.798218032,
                4508934.20108052,
                4430966.68254922,
                769700.190206802,
                4509638.17832858,
                4430266.50586499,
                769578.562994134,
                4510342.04561298,
                4429566.21869423,
                769456.916588924,
                4511045.80288106,
                4428865.82108818,
                769335.251000076,
                4511749.45008019,
                4428165.31309812,
                769213.566236498,
                4512452.98715773,
                4427464.69477534,
                769091.862307099,
                4513156.41406109,
                4426763.96617118,
                768970.139220795,
                4513859.73073772,
                4426063.12733698,
                768848.396986506,
                4514562.93713505,
                4425362.1783241,
                768726.635613157,
                4515266.03320058,
                4424661.11918396,
                768604.855109674,
                4515969.0188818,
                4423959.94996796,
                768483.055484991,
                4516671.89412624,
                4423258.67072757,
                768361.236748044,
                4517374.65888147,
                4422557.28151424,
                768239.398907774,
                4518077.31309504,
                4421855.78237947,
                768117.541973127,
                4518779.85671457,
                4421154.17337478,
                767995.665953051,
                4519482.28968768,
                4420452.45455172,
                767873.7708565,
                4520184.61196203,
                4419750.62596185,
                767751.856692433,
                4520886.82348528,
                4419048.68765677,
                767629.923469811,
                4521588.92420515,
                4418346.63968809,
                767507.9711976,
                4522290.91406935,
                4417644.48210745,
                767385.999884771,
                4522992.79302563,
                4416942.21496652,
                767264.009540299,
                4523694.56102177,
                4416239.83831699,
                767142.000173162,
                4524396.21800557,
                4415537.35221058,
                767019.971792345,
                4525097.76392485,
                4414834.75669901,
                766897.924406834,
                4525799.19872746,
                4414132.05183405,
                766775.85802562,
                4526500.52236127,
                4413429.2376675,
                766653.772657701,
                4527201.73477418,
                4412726.31425115,
                766531.668312075,
                4527902.83591412,
                4412023.28163684,
                766409.544997746,
                4528603.82572902,
                4411320.13987644,
                766287.402723724,
                4529304.70416687,
                4410616.88902183,
                766165.241499021,
                4530005.47117566,
                4409913.52912491,
                766043.061332653,
                4530706.12670341,
                4409210.06023761,
                765920.862233641,
                4531406.67069817,
                4408506.48241189,
                765798.644211011,
                4532107.10310801,
                4407802.79569973,
                765676.407273792,
                4532807.42388102,
                4407099.00015313,
                765554.151431016,
                4533507.63296533,
                4406395.09582412,
                765431.876691723,
                4534207.7303091,
                4405691.08276475,
                765309.583064953,
                4534907.71586047,
                4404986.96102709,
                765187.270559752,
                4535607.58956766,
                4404282.73066325,
                765064.939185171,
                4536307.35137889,
                4403578.39172534,
                764942.588950264,
                4537007.00124239,
                4402873.94426552,
                764820.219864089,
                4537706.53910645,
                4402169.38833596,
                764697.831935709,
                4538405.96491936,
                4401464.72398884,
                764575.42517419,
                4539105.27862944,
                4400759.95127639,
                764452.999588603,
                4539804.48018504,
                4400055.07025084,
                764330.555188024,
                4540503.56953453,
                4399350.08096448,
                764208.091981531,
                4541202.54662631,
                4398644.98346958,
                764085.609978207,
                4541901.4114088,
                4397939.77781846,
                763963.10918714,
                4542600.16383045,
                4397234.46406345,
                763840.589617422,
                4543298.80383973,
                4396529.04225692,
                763718.051278147,
                4543997.33138513,
                4395823.51245126,
                763595.494178416,
                4544695.74641519,
                4395117.87469886,
                763472.918327332,
                4545394.04887845,
                4394412.12905217,
                763350.323734004,
                4546092.23872348,
                4393706.27556363,
                763227.710407543,
                4546790.31589889,
                4393000.31428573,
                763105.078357066,
                4547488.28035329,
                4392294.24527097,
                762982.427591693,
                4548186.13203534,
                4391588.06857188,
                762859.758120548,
                4548883.87089371,
                4390881.78424101,
                762737.069952759,
                4549581.4968771,
                4390175.39233092,
                762614.363097459,
                4550279.00993424,
                4389468.89289422,
                762491.637563785,
                4550976.41001388,
                4388762.28598353,
                762368.893360877,
                4551673.6970648,
                4388055.57165149,
                762246.130497881,
                4552370.87103579,
                4387348.74995077,
                762123.348983944,
                4553067.93187569,
                4386641.82093406,
                762000.548828221,
                4553764.87953335,
                4385934.78465408,
                761877.730039867,
                4554461.71395765,
                4385227.64116356,
                761754.892628044,
                4555158.43509749,
                4384520.39051526,
                761632.036601917,
                4555855.04290179,
                4383813.03276197,
                761509.161970655,
                4556551.53731953,
                4383105.5679565,
                761386.268743432,
                4557247.91829967,
                4382397.99615167,
                761263.356929424,
                4557944.18579122,
                4381690.31740035,
                761140.426537814,
                4558640.33974322,
                4380982.5317554,
                761017.477577785,
                4559336.38010472,
                4380274.63926974,
                760894.510058529,
                4560032.30682481,
                4379566.63999628,
                760771.523989238,
                4560728.11985259,
                4378858.53398798,
                760648.519379109,
                4561423.8191372,
                4378150.3212978,
                760525.496237345,
                4562119.4046278,
                4377442.00197873,
                760402.45457315,
                4562814.87627357,
                4376733.5760838,
                760279.394395735,
                4563510.23402374,
                4376025.04366604,
                760156.315714312,
                4564205.47782752,
                4375316.40477852,
                760033.2185381,
                4564900.6076342,
                4374607.65947433,
                759910.10287632,
                4565595.62339305,
                4373898.80780657,
                759786.968738198,
                4566290.52505339,
                4373189.84982837,
                759663.816132963,
                4566985.31256457,
                4372480.7855929,
                759540.645069849,
                4567679.98587594,
                4371771.61515333,
                759417.455558094,
                4568374.54493691,
                4371062.33856286,
                759294.247606939,
                4569068.98969688,
                4370352.95587472,
                759171.021225629,
                4569763.32010531,
                4369643.46714215,
                759047.776423416,
                4570457.53611167,
                4368933.87241843,
                758924.513209551,
                4571151.63766544,
                4368224.17175684,
                758801.231593292,
                4571845.62471616,
                4367514.36521071,
                758677.931583902,
                4572539.49721336,
                4366804.45283338,
                758554.613190646,
                4573233.25510664,
                4366094.4346782,
                758431.276422792,
                4573926.89834557,
                4365384.31079856,
                758307.921289616,
                4574620.4268798,
                4364674.08124787,
                758184.547800393,
                4575313.84065897,
                4363963.74607956,
                758061.155964406,
                4576007.13963276,
                4363253.30534709,
                757937.745790939,
                4576700.32375087,
                4362542.75910392,
                757814.317289283,
                4577393.39296305,
                4361832.10740356,
                757690.87046873,
                4578086.34721903,
                4361121.35029953,
                757567.405338578,
                4578779.18646861,
                4360410.48784538,
                757443.921908127,
                4579471.91066159,
                4359699.52009466,
                757320.420186683,
                4580164.51974781,
                4358988.44710098,
                757196.900183555,
                4580857.01367713,
                4358277.26891794,
                757073.361908056,
                4581549.39239944,
                4357565.98559918,
                756949.805369502,
                4582241.65586465,
                4356854.59719835,
                756826.230577215,
                4582933.8040227,
                4356143.10376914,
                756702.637540519,
                4583625.83682356,
                4355431.50536525,
                756579.026268743,
                4584317.75421722,
                4354719.80204039,
                756455.396771219,
                4585009.55615369,
                4354007.99384834,
                756331.749057284,
                4585701.24258303,
                4353296.08084284,
                756208.083136278,
                4586392.81345531,
                4352584.06307769,
                756084.399017546,
                4587084.26872061,
                4351871.94060671,
                755960.696710435,
                4587775.60832908,
                4351159.71348374,
                755836.976224299,
                4588466.83223085,
                4350447.38176264,
                755713.237568492,
                4589157.94037611,
                4349734.94549728,
                755589.480752374,
                4589848.93271505,
                4349022.40474158,
                755465.705785311,
                4590539.80919791,
                4348309.75954946,
                755341.912676667,
                4591230.56977495,
                4347597.00997487,
                755218.101435817,
                4591921.21439644,
                4346884.15607178,
                755094.272072135,
                4592611.7430127,
                4346171.19789418,
                754970.424595,
                4593302.15557406,
                4345458.1354961,
                754846.559013795,
                4593992.45203088,
                4344744.96893157,
                754722.675337907,
                4594682.63233356,
                4344031.69825465,
                754598.773576728,
                4595372.6964325,
                4343318.32351942,
                754474.853739652,
                4596062.64427816,
                4342604.84478,
                754350.915836077,
                4596752.475821,
                4341891.2620905,
                754226.959875406,
                4597442.19101151,
            ]
        )

        time_axis = RegularAxis(
            (0, 5, state_vectors.size // 3),
            PreciseDateTime.from_utc_string("17-FEB-2020 16:00:34.908999712209"),
        )
        orbit = GeneralSarOrbit(time_axis, state_vectors)
        self.trajectory = GSO3DCurveWrapper(orbit=orbit)

        self.range_times = np.linspace(0.004367891, 0.00451789, 100)
        self.az_time = time_axis.get_array()[25]
        self.look_dir = "RIGHT"

        self.tolerance = 1e-8
        self.look_angles_expected = np.array(
            [
                1.51725199,
                1.51723499,
                1.51721799,
                1.51720098,
                1.51718398,
                1.51716698,
                1.51714998,
                1.51713297,
                1.51711597,
                1.51709896,
                1.51708196,
                1.51706495,
                1.51704795,
                1.51703094,
                1.51701393,
                1.51699692,
                1.51697991,
                1.5169629,
                1.51694589,
                1.51692888,
                1.51691187,
                1.51689486,
                1.51687785,
                1.51686084,
                1.51684382,
                1.51682681,
                1.51680979,
                1.51679278,
                1.51677576,
                1.51675875,
                1.51674173,
                1.51672471,
                1.5167077,
                1.51669068,
                1.51667366,
                1.51665664,
                1.51663962,
                1.5166226,
                1.51660558,
                1.51658856,
                1.51657153,
                1.51655451,
                1.51653749,
                1.51652046,
                1.51650344,
                1.51648642,
                1.51646939,
                1.51645236,
                1.51643534,
                1.51641831,
                1.51640128,
                1.51638426,
                1.51636723,
                1.5163502,
                1.51633317,
                1.51631614,
                1.51629911,
                1.51628208,
                1.51626504,
                1.51624801,
                1.51623098,
                1.51621394,
                1.51619691,
                1.51617988,
                1.51616284,
                1.51614581,
                1.51612877,
                1.51611173,
                1.5160947,
                1.51607766,
                1.51606062,
                1.51604358,
                1.51602654,
                1.5160095,
                1.51599246,
                1.51597542,
                1.51595838,
                1.51594134,
                1.51592429,
                1.51590725,
                1.51589021,
                1.51587316,
                1.51585612,
                1.51583907,
                1.51582203,
                1.51580498,
                1.51578794,
                1.51577089,
                1.51575384,
                1.51573679,
                1.51571974,
                1.51570269,
                1.51568564,
                1.51566859,
                1.51565154,
                1.51563449,
                1.51561744,
                1.51560039,
                1.51558333,
                1.51556628,
            ]
        )
        self.incidence_angles_expected = np.array(
            [
                1.62010951,
                1.62012822,
                1.62014693,
                1.62016564,
                1.62018435,
                1.62020306,
                1.62022177,
                1.62024047,
                1.62025918,
                1.62027788,
                1.62029659,
                1.62031529,
                1.620334,
                1.6203527,
                1.62037141,
                1.62039011,
                1.62040881,
                1.62042751,
                1.62044621,
                1.62046491,
                1.62048361,
                1.62050231,
                1.62052101,
                1.62053971,
                1.62055841,
                1.62057711,
                1.6205958,
                1.6206145,
                1.62063319,
                1.62065189,
                1.62067058,
                1.62068928,
                1.62070797,
                1.62072667,
                1.62074536,
                1.62076405,
                1.62078274,
                1.62080143,
                1.62082012,
                1.62083881,
                1.6208575,
                1.62087619,
                1.62089488,
                1.62091357,
                1.62093226,
                1.62095094,
                1.62096963,
                1.62098832,
                1.621007,
                1.62102569,
                1.62104437,
                1.62106306,
                1.62108174,
                1.62110042,
                1.6211191,
                1.62113779,
                1.62115647,
                1.62117515,
                1.62119383,
                1.62121251,
                1.62123119,
                1.62124987,
                1.62126855,
                1.62128722,
                1.6213059,
                1.62132458,
                1.62134325,
                1.62136193,
                1.62138061,
                1.62139928,
                1.62141795,
                1.62143663,
                1.6214553,
                1.62147398,
                1.62149265,
                1.62151132,
                1.62152999,
                1.62154866,
                1.62156733,
                1.621586,
                1.62160467,
                1.62162334,
                1.62164201,
                1.62166068,
                1.62167934,
                1.62169801,
                1.62171668,
                1.62173534,
                1.62175401,
                1.62177267,
                1.62179134,
                1.62181,
                1.62182867,
                1.62184733,
                1.62186599,
                1.62188466,
                1.62190332,
                1.62192198,
                1.62194064,
                1.6219593,
            ]
        )

    def test_compute_look_angles_from_trajectory_case0(self) -> None:
        """Testing compute_look_angles_from_trajectory function, case 0"""

        # case 0: single range value
        angles = compute_look_angles_from_trajectory(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times[0],
            look_direction=self.look_dir,
        )

        # checking results
        self.assertIsInstance(angles, float)
        np.testing.assert_allclose(
            angles, self.look_angles_expected[0], atol=self.tolerance, rtol=0
        )

    def test_compute_look_angles_from_trajectory_case1(self) -> None:
        """Testing compute_look_angles_from_trajectory function, case 1"""

        # case 1: range array
        angles = compute_look_angles_from_trajectory(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times,
            look_direction=self.look_dir,
        )

        # checking results
        self.assertIsInstance(angles, np.ndarray)
        self.assertEqual(angles.ndim, 1)
        self.assertEqual(angles.size, self.look_angles_expected.size)
        np.testing.assert_allclose(
            angles, self.look_angles_expected, atol=self.tolerance, rtol=0
        )

    def test_compute_incidence_angles_from_trajectory_case0(self) -> None:
        """Testing compute_incidence_angles_from_trajectory function, case 0"""

        # case 0: single range value
        angles = compute_incidence_angles_from_trajectory(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times[0],
            look_direction=self.look_dir,
        )

        # checking results
        self.assertIsInstance(angles, float)
        np.testing.assert_allclose(
            angles, self.incidence_angles_expected[0], atol=self.tolerance, rtol=0
        )

    def test_compute_incidence_angles_from_trajectory_case1(self) -> None:
        """Testing compute_incidence_angles_from_trajectory function, case 1"""

        # case 1: range array
        angles = compute_incidence_angles_from_trajectory(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times,
            look_direction=self.look_dir,
        )

        # checking results
        self.assertIsInstance(angles, np.ndarray)
        self.assertEqual(angles.ndim, 1)
        self.assertEqual(angles.size, self.incidence_angles_expected.size)
        np.testing.assert_allclose(
            angles, self.incidence_angles_expected, atol=self.tolerance, rtol=0
        )


class GeometricSquintTest(TestCase):
    """Testing get_geometric_squint function"""

    def setUp(self):
        self.pos_0 = np.array(
            [-2449675.14554249, -5216814.136353868, 3907089.2898868835]
        )
        self.pos_20 = np.array(
            [4397940.093636902, 763963.1640477455, 4542599.8509511445]
        )
        self.vel_0 = np.array(
            [-3283.937062880771, -3101.7802725409233, -6163.652047976267]
        )
        self.vel_20 = np.array(
            [-141.05193267576627, -24.50203469983142, 139.73925487802535]
        )
        self.point_0 = np.array(
            [-2467483.2210648037, -4626385.185907534, 3619451.3347408967]
        )
        self.point_20 = np.array(
            [4397211.556651601, 766361.9969266983, 4540802.84326352]
        )

        self.squint_ref_0 = -5.853944778390092e-11
        self.squint_ref_20 = -0.34229904620461094

        self.tolerance = 1e-16

    def test_get_geometric_squint_zero(self) -> None:
        """Testing getting an almost 0 squint angle"""
        squint = get_geometric_squint(
            sensor_positions=self.pos_0,
            sensor_velocities=self.vel_0,
            ground_points=self.point_0,
        )
        self.assertIsInstance(squint, float)
        np.testing.assert_allclose(
            squint, self.squint_ref_0, atol=self.tolerance, rtol=0
        )

    def test_get_geometric_squint_non_zero(self) -> None:
        """Testing getting a squint angle different from 0"""
        squint = get_geometric_squint(
            sensor_positions=self.pos_20,
            sensor_velocities=self.vel_20,
            ground_points=self.point_20,
        )
        self.assertIsInstance(squint, float)
        np.testing.assert_allclose(
            squint, self.squint_ref_20, atol=self.tolerance, rtol=0
        )

    def test_get_multiple_squint(self) -> None:
        """Testing function vectorization"""
        squints = get_geometric_squint(
            sensor_positions=np.vstack([self.pos_0, self.pos_20]),
            sensor_velocities=np.vstack([self.vel_0, self.vel_20]),
            ground_points=np.vstack([self.point_0, self.point_20]),
        )
        self.assertIsInstance(squints, np.ndarray)
        self.assertTrue(squints.ndim == 1)
        self.assertTrue(squints.size == 2)
        np.testing.assert_allclose(
            squints,
            np.array([self.squint_ref_0, self.squint_ref_20]),
            atol=self.tolerance,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
