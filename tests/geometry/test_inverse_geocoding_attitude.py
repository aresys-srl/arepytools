# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing arepytools/geometry/inverse_geocoding inverse_geocoding_attitude functionalities"""

import unittest

import numpy as np

from arepytools.geometry.generalsarattitude import (
    GeneralSarAttitude,
    create_attitude_boresight_normal_curve_wrapper,
)
from arepytools.geometry.generalsarorbit import GeneralSarOrbit, GSO3DCurveWrapper
from arepytools.geometry.inverse_geocoding import inverse_geocoding_attitude
from arepytools.geometry.inverse_geocoding_core import (
    AmbiguousInputCorrelation,
    inverse_geocoding_attitude_core,
)
from arepytools.math.axis import RegularAxis
from arepytools.timing.precisedatetime import PreciseDateTime

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


class InverseGeocodingAttitudeCoreTest(unittest.TestCase):
    """Testing inverse geocoding attitude core"""

    def setUp(self):
        _yaw = np.array([-10.9353126825047] * 160)
        _pitch = np.array([-19.9452555208155] * 160)
        _roll = np.array([-57.1252699331354] * 160)

        _ypr_matrix = np.vstack((_yaw, _pitch, _roll))

        # creating orbit and orbit curve wrapper
        self.orbit = GeneralSarOrbit(time_axis, state_vectors)
        self.gso_curve = GSO3DCurveWrapper(orbit=self.orbit)

        # creating attitude and attitude curve wrapper
        self.attitude = GeneralSarAttitude(
            self.orbit, time_axis, _ypr_matrix, "RPY", "ZERODOPPLER"
        )
        self.arf_boresight_normal_curve = (
            create_attitude_boresight_normal_curve_wrapper(attitude=self.attitude)
        )

        self.t_ref = PreciseDateTime.from_utc_string(
            "17-FEB-2020 16:06:59.908999712209"
        )

        self.ground_points = np.array(
            [
                [4397211.561197397, 766361.9958713502, 4540802.839067862],
                [4397221.561197397, 766371.9958713502, 4540812.839067862],
                [4397231.561197397, 766381.9958713502, 4540822.839067862],
                [4397241.561197397, 766391.9958713502, 4540832.839067862],
            ]
        )

        self.tolerance = 1e-10
        self.rng_abs_tolerance = 1e-17
        self.az_abs_tolerance = 1e-10
        az_ref = [
            "17-FEB-2020 16:07:14.906728901874",
            "17-FEB-2020 16:07:14.879990037744",
            "17-FEB-2020 16:07:14.853251305845",
            "17-FEB-2020 16:07:14.826512706175",
        ]
        self.az_results_ref = list(map(PreciseDateTime.from_utc_string, az_ref))
        self.init_guess = self.az_results_ref[0]
        self.rng_results_ref = np.array(
            [
                2.0577841466333684e-05,
                2.056345789857485e-05,
                2.05497461246835e-05,
                2.053670749001785e-05,
            ]
        )

    def test_inverse_geocoding_attitude_core_case0a(self) -> None:
        """Testing inverse_geocoding_attitude_core from geometry/inverse_geocoding.py submodule, case 0a"""

        # case 0a: 1 earth points (3,), 1 init guess PDT
        az_times, rng_times = inverse_geocoding_attitude_core(
            trajectory=self.gso_curve,
            boresight_normal=self.arf_boresight_normal_curve,
            initial_guesses=self.t_ref + 10,
            ground_points=self.ground_points[0],
        )

        # checking results
        self.assertTrue(isinstance(az_times, PreciseDateTime))
        self.assertTrue(isinstance(rng_times, float))

        self.assertTrue(
            np.abs(az_times - self.az_results_ref[0]) < self.az_abs_tolerance
        )
        self.assertTrue(
            np.abs(rng_times - self.rng_results_ref[0]) < self.rng_abs_tolerance
        )

    def test_inverse_geocoding_attitude_core_case0b(self) -> None:
        """Testing inverse_geocoding_attitude_core from geometry/inverse_geocoding.py submodule, case 0b"""

        # case 0b: 1 earth points (1, 3), 1 init guess PDT
        az_times, rng_times = inverse_geocoding_attitude_core(
            trajectory=self.gso_curve,
            boresight_normal=self.arf_boresight_normal_curve,
            initial_guesses=self.t_ref + 10,
            ground_points=self.ground_points[0].reshape(1, 3),
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == 1)
        self.assertTrue(rng_times.size == 1)

        delta_az = np.array(az_times - np.array(self.az_results_ref[0]), dtype=float)
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times,
            np.array([self.rng_results_ref[0]]),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_attitude_core_case0c(self) -> None:
        """Testing inverse_geocoding_attitude_core from geometry/inverse_geocoding.py submodule, case 0c"""

        # case 0c: 1 earth points (1, 3), 1 init guess (array)
        az_times, rng_times = inverse_geocoding_attitude_core(
            trajectory=self.gso_curve,
            boresight_normal=self.arf_boresight_normal_curve,
            initial_guesses=np.array([self.t_ref + 10]),
            ground_points=self.ground_points[0].reshape(1, 3),
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == 1)
        self.assertTrue(rng_times.size == 1)

        delta_az = np.array(az_times - np.array(self.az_results_ref[0]), dtype=float)
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times,
            np.array([self.rng_results_ref[0]]),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_attitude_core_case1(self) -> None:
        """Testing inverse_geocoding_attitude_core from geometry/inverse_geocoding.py submodule, case 1"""

        # case 1: N earth points (N, 3), 1 init guess PDT
        az_times, rng_times = inverse_geocoding_attitude_core(
            trajectory=self.gso_curve,
            boresight_normal=self.arf_boresight_normal_curve,
            initial_guesses=self.t_ref + 10,
            ground_points=self.ground_points,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.ground_points.shape[0])
        self.assertTrue(rng_times.size == self.ground_points.shape[0])

        delta_az = np.array(az_times - np.array(self.az_results_ref), dtype=float)
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times, self.rng_results_ref, atol=self.rng_abs_tolerance, rtol=0
        )

    def test_inverse_geocoding_attitude_core_case2(self) -> None:
        """Testing inverse_geocoding_attitude_core from geometry/inverse_geocoding.py submodule, case 2"""

        # case 2: N earth points (N, 3), N init guess (N,)
        az_times, rng_times = inverse_geocoding_attitude_core(
            trajectory=self.gso_curve,
            boresight_normal=self.arf_boresight_normal_curve,
            initial_guesses=np.repeat(self.t_ref + 10, self.ground_points.shape[0]),
            ground_points=self.ground_points,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.ground_points.shape[0])
        self.assertTrue(rng_times.size == self.ground_points.shape[0])

        delta_az = np.array(az_times - np.array(self.az_results_ref), dtype=float)
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times, self.rng_results_ref, atol=self.rng_abs_tolerance, rtol=0
        )

    def test_inverse_geocoding_attitude_core_case3a(self) -> None:
        """Testing inverse_geocoding_attitude_core from geometry/inverse_geocoding.py submodule, case 3a"""

        # case 3a: m_rip guesses, 1 earth point (3,)
        m_rip = 5
        az_times, rng_times = inverse_geocoding_attitude_core(
            trajectory=self.gso_curve,
            boresight_normal=self.arf_boresight_normal_curve,
            initial_guesses=np.repeat(self.t_ref + 10, m_rip),
            ground_points=self.ground_points[0],
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == m_rip)
        self.assertTrue(rng_times.size == m_rip)

        delta_az = np.array(
            az_times - np.repeat(self.az_results_ref[0], m_rip), dtype=float
        )
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.rng_results_ref[0], m_rip),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_attitude_core_case3b(self) -> None:
        """Testing inverse_geocoding_attitude_core from geometry/inverse_geocoding.py submodule, case 3b"""

        # case 3b: m_rip guesses, 1 earth point (1, 3)
        m_rip = 5
        az_times, rng_times = inverse_geocoding_attitude_core(
            trajectory=self.gso_curve,
            boresight_normal=self.arf_boresight_normal_curve,
            initial_guesses=np.repeat(self.t_ref + 10, m_rip),
            ground_points=self.ground_points[0].reshape(1, 3),
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == m_rip)
        self.assertTrue(rng_times.size == m_rip)

        delta_az = np.array(
            az_times - np.repeat(self.az_results_ref[0], m_rip), dtype=float
        )
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.rng_results_ref[0], m_rip),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )


class InverseGeocodingAttitudeTest(unittest.TestCase):
    """Testing inverse geocoding with attitude"""

    def setUp(self):
        _yaw = np.array([-10.9353126825047] * 160)
        _pitch = np.array([-19.9452555208155] * 160)
        _roll = np.array([-57.1252699331354] * 160)

        _ypr_matrix = np.vstack((_yaw, _pitch, _roll))

        # creating orbit and orbit curve wrapper
        self.orbit = GeneralSarOrbit(time_axis, state_vectors)
        self.gso_curve = GSO3DCurveWrapper(orbit=self.orbit)

        # creating attitude and attitude curve wrapper
        self.attitude = GeneralSarAttitude(
            self.orbit, time_axis, _ypr_matrix, "RPY", "ZERODOPPLER"
        )
        self.arf_boresight_normal_curve = (
            create_attitude_boresight_normal_curve_wrapper(attitude=self.attitude)
        )

        self.t_ref = PreciseDateTime.from_utc_string(
            "17-FEB-2020 16:06:59.908999712209"
        )

        self.ground_points = np.array(
            [
                [4397211.561197397, 766361.9958713502, 4540802.839067862],
                [4397221.561197397, 766371.9958713502, 4540812.839067862],
                [4397231.561197397, 766381.9958713502, 4540822.839067862],
                [4397241.561197397, 766391.9958713502, 4540832.839067862],
            ]
        )

        self.tolerance = 1e-10
        self.rng_abs_tolerance = 1e-17
        self.az_abs_tolerance = 1e-10
        az_ref = [
            "17-FEB-2020 16:07:14.906728901874",
            "17-FEB-2020 16:07:14.879990037744",
            "17-FEB-2020 16:07:14.853251305845",
            "17-FEB-2020 16:07:14.826512706175",
        ]
        self.az_results_ref = list(map(PreciseDateTime.from_utc_string, az_ref))
        self.init_guess = self.az_results_ref[0]
        self.rng_results_ref = np.array(
            [
                2.0577841466333684e-05,
                2.056345789857485e-05,
                2.05497461246835e-05,
                2.053670749001785e-05,
            ]
        )

    def test_inverse_geocoding_attitude_case0a(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 0a"""

        # case 0a: 1 point (3,)
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points[0],
        )

        # checking results
        self.assertTrue(isinstance(az_times, PreciseDateTime))
        self.assertTrue(isinstance(rng_times, float))

        self.assertTrue(
            np.abs(az_times - self.az_results_ref[0]) < self.az_abs_tolerance
        )
        self.assertTrue(
            np.abs(rng_times - self.rng_results_ref[0]) < self.rng_abs_tolerance
        )

    def test_inverse_geocoding_attitude_case0b(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 0b"""

        # case 0b: 1 point (1, 3)
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points[0].reshape(1, 3),
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)

        delta_az = np.array(az_times - np.array([self.az_results_ref[0]]), dtype=float)
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times,
            np.array([self.rng_results_ref[0]]),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_attitude_case0c(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 0c"""

        # case 0c: 1 point (3,), 1 init guess PDT
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points[0],
            az_initial_time_guesses=self.init_guess,
        )

        # checking results
        self.assertTrue(isinstance(az_times, PreciseDateTime))
        self.assertTrue(isinstance(rng_times, float))

        self.assertTrue(
            np.abs(az_times - self.az_results_ref[0]) < self.az_abs_tolerance
        )
        self.assertTrue(
            np.abs(rng_times - self.rng_results_ref[0]) < self.rng_abs_tolerance
        )

    def test_inverse_geocoding_attitude_case0d(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 0d"""

        # case 0d: 1 point (1, 3), 1 init guess PDT
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points[0].reshape(1, 3),
            az_initial_time_guesses=self.init_guess,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == 1)
        self.assertTrue(rng_times.size == 1)

        delta_az = np.array(
            az_times - np.array([self.az_results_ref[0]] * 2), dtype=float
        )
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times,
            np.array([self.rng_results_ref[0]]),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_attitude_case1a(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 1a"""

        # case 1a: N points (N, 3)
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.ground_points.shape[0])
        self.assertTrue(rng_times.size == self.ground_points.shape[0])

        delta_az = np.array(az_times - self.az_results_ref, dtype=float)
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times, self.rng_results_ref, atol=self.rng_abs_tolerance, rtol=0
        )

    def test_inverse_geocoding_attitude_case1b(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 1b"""

        # case 1b: N points (N, 3), 1 init guess PDT
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points,
            az_initial_time_guesses=self.init_guess,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.ground_points.shape[0])
        self.assertTrue(rng_times.size == self.ground_points.shape[0])

        delta_az = np.array(az_times - self.az_results_ref, dtype=float)
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times, self.rng_results_ref, atol=self.rng_abs_tolerance, rtol=0
        )

    def test_inverse_geocoding_attitude_case1c(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 1c"""

        # case 1c: N points (N, 3), N init guess (N,)
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points,
            az_initial_time_guesses=np.repeat(
                self.init_guess, self.ground_points.shape[0]
            ),
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.ground_points.shape[0])
        self.assertTrue(rng_times.size == self.ground_points.shape[0])

        delta_az = np.array(az_times - self.az_results_ref, dtype=float)
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times, self.rng_results_ref, atol=self.rng_abs_tolerance, rtol=0
        )

    def test_inverse_geocoding_attitude_case2a(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 2a"""

        # case 2a: 1 point (3,), m_rip init guess (m_rip,)
        m_rip = 5
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points[0],
            az_initial_time_guesses=np.repeat(self.init_guess, m_rip),
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == m_rip)
        self.assertTrue(rng_times.size == m_rip)

        delta_az = np.array(
            az_times - np.repeat(self.az_results_ref[0], m_rip), dtype=float
        )
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.rng_results_ref[0], m_rip),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_attitude_case2b(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 2b"""

        # case 2b: 1 point (1,3), m_rip init guess (m_rip,)
        m_rip = 5
        az_times, rng_times = inverse_geocoding_attitude(
            orbit=self.orbit,
            attitude=self.attitude,
            ground_points=self.ground_points[0].reshape(1, 3),
            az_initial_time_guesses=np.repeat(self.init_guess, m_rip),
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == m_rip)
        self.assertTrue(rng_times.size == m_rip)

        delta_az = np.array(
            az_times - np.repeat(self.az_results_ref[0], m_rip), dtype=float
        )
        np.testing.assert_allclose(
            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.rng_results_ref[0], m_rip),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_attitude_case3(self) -> None:
        """Testing inverse_geocoding_attitude from geometry/inverse_geocoding.py submodule, case 3"""

        # case 3: N points, M PDT RAISES ERROR BY DESIGN, NOT SUPPORTED
        with self.assertRaises(AmbiguousInputCorrelation):
            inverse_geocoding_attitude(
                orbit=self.orbit,
                attitude=self.attitude,
                ground_points=self.ground_points,
                az_initial_time_guesses=np.repeat(self.init_guess, 2),
            )


if __name__ == "__main__":
    unittest.main()
