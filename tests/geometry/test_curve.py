# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing arepytools/geometry/curve module"""

import unittest
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import numpy.typing as npt

import arepytools.geometry.curve as curve_3d
from arepytools.geometry.generalsarattitude import (
    GeneralSarAttitude,
    create_attitude_boresight_normal_curve_wrapper,
)
from arepytools.geometry.generalsarorbit import GeneralSarOrbit, GSO3DCurveWrapper
from arepytools.math.axis import RegularAxis
from arepytools.timing.precisedatetime import PreciseDateTime


@dataclass
class TestingBundle:
    """Testing bundle dataclass"""

    method: str = None
    curve: Union[
        curve_3d.SplineWrapper, curve_3d.PolynomialWrapper, curve_3d.Generic3DCurve
    ] = None
    params: Union[float, npt.ArrayLike, PreciseDateTime] = None
    expected_output: Union[float, np.ndarray] = None


def _bundle_testing(
    bundle: List[TestingBundle], shape: tuple, tolerance: float
) -> None:
    """Custom function to wrap the bundle testing in one place.

    Parameters
    ----------
    bundle : List[TestingBundle]
        input list of Testing Bundle dataclasses
    shape : tuple
        shape to be checked for outputs of this test
    tolerance : tuple
        tolerance for values checking
    """
    for item in bundle:
        if item.method == "eval":
            res = item.curve.evaluate(item.params)
        elif item.method == "eval_der1":
            res = item.curve.evaluate_first_derivative(item.params)
        elif item.method == "eval_der2":
            res = item.curve.evaluate_second_derivative(item.params)

        assert np.shape(res) == shape
        np.testing.assert_allclose(res, item.expected_output, atol=tolerance, rtol=0)


def _bundle_testing_3d(
    bundle: List[TestingBundle], shape: tuple, tolerance: float
) -> None:
    """Custom function to wrap the bundle testing in one place.

    Parameters
    ----------
    bundle : List[TestingBundle]
        input list of TestingBundle dataclasses
    shape : tuple
        shape to be checked for outputs of this test
    tolerance : tuple
        tolerance for values checking
    """
    for item in bundle:
        if item.method == "eval":
            res = item.curve.evaluate(item.params)
        elif item.method == "eval_der1":
            res = item.curve.evaluate_first_derivatives(item.params)
        elif item.method == "eval_der2":
            res = item.curve.evaluate_second_derivatives(item.params)

        assert np.shape(res) == shape
        np.testing.assert_allclose(res, item.expected_output, atol=tolerance, rtol=0)


class Generic3DCurveTest(unittest.TestCase):
    """Testing Generic3DCurve class"""

    def setUp(self):
        self.x_func = curve_3d.PolynomialWrapper(coeff=[1, 2, 3, 4, 5])
        self.y_func = curve_3d.PolynomialWrapper(coeff=[1, 2, 3, 4, 5])
        self.z_func = curve_3d.PolynomialWrapper(coeff=[1, 2, 3, 4, 5])
        self.t_ref = PreciseDateTime.from_utc_string(
            "17-FEB-2020 16:06:59.908999712209"
        )
        t_bound = (self.t_ref, self.t_ref + 30)
        self.curve = curve_3d.Generic3DCurve(
            self.x_func, self.y_func, self.z_func, self.t_ref, t_bound
        )
        # expected results
        eval_res = 12345
        eval_der1_res = 4664
        eval_der2_res = 1326

        # setting up test bundles
        self.curve_testing_scalar_bundle = {
            "input": self.t_ref + 10,
            "eval": np.repeat(eval_res, 3),
            "eval_der1": np.repeat(eval_der1_res, 3),
            "eval_der2": np.repeat(eval_der2_res, 3),
        }
        self.curve_testing_array_bundle = {
            "input": np.repeat(self.t_ref + 10, 4),
            "eval": np.full((4, 3), eval_res),
            "eval_der1": np.full((4, 3), eval_der1_res),
            "eval_der2": np.full((4, 3), eval_der2_res),
        }
        self.curve_testing_array1_bundle = {
            "input": np.array([self.t_ref + 10]),
            "eval": np.repeat(eval_res, 3).reshape(1, 3),
            "eval_der1": np.repeat(eval_der1_res, 3).reshape(1, 3),
            "eval_der2": np.repeat(eval_der2_res, 3).reshape(1, 3),
        }

        self.tolerance = 1e-16

    def test_generic_sar_function_methods_on_scalar_input(self) -> None:
        """Testing GenericSarDifferentiableFunction methods on scalar input"""
        out = self.curve.evaluate(self.curve_testing_scalar_bundle["input"])
        out_der1 = self.curve.evaluate_first_derivatives(
            self.curve_testing_scalar_bundle["input"]
        )
        out_der2 = self.curve.evaluate_second_derivatives(
            self.curve_testing_scalar_bundle["input"]
        )

        self.assertTrue(np.shape(out) == (3,))
        self.assertTrue(np.shape(out_der1) == (3,))
        self.assertTrue(np.shape(out_der2) == (3,))
        np.testing.assert_allclose(
            out, self.curve_testing_scalar_bundle["eval"], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            out_der1,
            self.curve_testing_scalar_bundle["eval_der1"],
            atol=self.tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            out_der2,
            self.curve_testing_scalar_bundle["eval_der2"],
            atol=self.tolerance,
            rtol=0,
        )

    def test_generic_sar_function_methods_on_array_input(self) -> None:
        """Testing GenericSarDifferentiableFunction methods on array input"""
        out = self.curve.evaluate(self.curve_testing_array_bundle["input"])
        out_der1 = self.curve.evaluate_first_derivatives(
            self.curve_testing_array_bundle["input"]
        )
        out_der2 = self.curve.evaluate_second_derivatives(
            self.curve_testing_array_bundle["input"]
        )

        self.assertTrue(np.shape(out) == (4, 3))
        self.assertTrue(np.shape(out_der1) == (4, 3))
        self.assertTrue(np.shape(out_der2) == (4, 3))
        np.testing.assert_allclose(
            out, self.curve_testing_array_bundle["eval"], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            out_der1,
            self.curve_testing_array_bundle["eval_der1"],
            atol=self.tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            out_der2,
            self.curve_testing_array_bundle["eval_der2"],
            atol=self.tolerance,
            rtol=0,
        )

    def test_generic_sar_function_methods_on_array1_input(self) -> None:
        """Testing GenericSarDifferentiableFunction methods on array input of size 1"""
        out = self.curve.evaluate(self.curve_testing_array1_bundle["input"])
        out_der1 = self.curve.evaluate_first_derivatives(
            self.curve_testing_array1_bundle["input"]
        )
        out_der2 = self.curve.evaluate_second_derivatives(
            self.curve_testing_array1_bundle["input"]
        )

        self.assertTrue(np.shape(out) == (1, 3))
        self.assertTrue(np.shape(out_der1) == (1, 3))
        self.assertTrue(np.shape(out_der2) == (1, 3))
        np.testing.assert_allclose(
            out, self.curve_testing_array1_bundle["eval"], atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            out_der1,
            self.curve_testing_array1_bundle["eval_der1"],
            atol=self.tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            out_der2,
            self.curve_testing_array1_bundle["eval_der2"],
            atol=self.tolerance,
            rtol=0,
        )

    def test_inverted_time_boundaries(self) -> None:
        """Testing GenericCurve3D generation with wrong time boundaries"""
        with self.assertRaises(curve_3d.InvertedTimeBoundaries):
            curve_3d.Generic3DCurve(
                self.x_func,
                self.y_func,
                self.z_func,
                self.t_ref,
                (self.t_ref + 5, self.t_ref - 5),
            )

    def test_time_out_of_interpolation_domain_case0a(self) -> None:
        """Testing GenericCurve3D evaluate time out of boundaries, case 0a"""
        with self.assertRaises(curve_3d.InterpolationOutOfBoundaries):
            self.curve.evaluate(self.t_ref - 10)

    def test_time_out_of_interpolation_domain_case0b(self) -> None:
        """Testing GenericCurve3D evaluate time out of boundaries, case 0b"""
        with self.assertRaises(curve_3d.InterpolationOutOfBoundaries):
            self.curve.evaluate(self.t_ref + 50)


class PolynomialSplineWrapperTest(unittest.TestCase):
    """Testing GenericSarDifferentiableFunction and GenericSarSpline classes"""

    def setUp(self):
        # polynomial curve from coefficients
        coefficients = [1, 2, 3, 4, 5]
        poly = curve_3d.PolynomialWrapper(coefficients)

        # spline curve
        axis = np.linspace(0, 200)
        values = axis**3 + 3 * axis**2 - 2 * axis + 6
        spline = curve_3d.SplineWrapper(axis, values)

        # inputs
        points = [1, 2, 3]
        # tolerance for results check
        self.tolerance = 1e-13

        # expected results poly
        poly_eval_res = [15, 57, 179]
        poly_eval_der1_res = [20, 72, 184]
        poly_eval_der2_res = [30, 78, 150]

        # expected results spline
        spline_eval_res = [8, 22, 54]
        spline_eval_der1_res = [7, 22, 43]
        spline_eval_der2_res = [12, 18, 24]

        # setup classes testing on scalar input
        self.testing_on_scalar_bundle = [
            TestingBundle(
                method="eval",
                curve=poly,
                params=points[0],
                expected_output=poly_eval_res[0],
            ),
            TestingBundle(
                method="eval",
                curve=spline,
                params=points[0],
                expected_output=spline_eval_res[0],
            ),
            TestingBundle(
                method="eval_der1",
                curve=poly,
                params=points[0],
                expected_output=poly_eval_der1_res[0],
            ),
            TestingBundle(
                method="eval_der1",
                curve=spline,
                params=points[0],
                expected_output=spline_eval_der1_res[0],
            ),
            TestingBundle(
                method="eval_der2",
                curve=poly,
                params=points[0],
                expected_output=poly_eval_der2_res[0],
            ),
            TestingBundle(
                method="eval_der2",
                curve=spline,
                params=points[0],
                expected_output=spline_eval_der2_res[0],
            ),
        ]

        # setup classes testing on array input
        self.testing_on_array_bundle = [
            TestingBundle(
                method="eval", curve=poly, params=points, expected_output=poly_eval_res
            ),
            TestingBundle(
                method="eval",
                curve=spline,
                params=points,
                expected_output=spline_eval_res,
            ),
            TestingBundle(
                method="eval_der1",
                curve=poly,
                params=points,
                expected_output=poly_eval_der1_res,
            ),
            TestingBundle(
                method="eval_der1",
                curve=spline,
                params=points,
                expected_output=spline_eval_der1_res,
            ),
            TestingBundle(
                method="eval_der2",
                curve=poly,
                params=points,
                expected_output=poly_eval_der2_res,
            ),
            TestingBundle(
                method="eval_der2",
                curve=spline,
                params=points,
                expected_output=spline_eval_der2_res,
            ),
        ]

        # setup classes testing on array input
        self.testing_on_array1_bundle = [
            TestingBundle(
                method="eval",
                curve=poly,
                params=np.array([points[0]]),
                expected_output=np.array([poly_eval_res[0]]),
            ),
            TestingBundle(
                method="eval",
                curve=spline,
                params=np.array([points[0]]),
                expected_output=np.array([spline_eval_res[0]]),
            ),
            TestingBundle(
                method="eval_der1",
                curve=poly,
                params=np.array([points[0]]),
                expected_output=np.array([poly_eval_der1_res[0]]),
            ),
            TestingBundle(
                method="eval_der1",
                curve=spline,
                params=np.array([points[0]]),
                expected_output=np.array([spline_eval_der1_res[0]]),
            ),
            TestingBundle(
                method="eval_der2",
                curve=poly,
                params=np.array([points[0]]),
                expected_output=np.array([poly_eval_der2_res[0]]),
            ),
            TestingBundle(
                method="eval_der2",
                curve=spline,
                params=np.array([points[0]]),
                expected_output=np.array([spline_eval_der2_res[0]]),
            ),
        ]

    def test_classes_on_scalar_input(self) -> None:
        """Testing instantiated class methods on scalar input"""
        _bundle_testing(
            bundle=self.testing_on_scalar_bundle, shape=(), tolerance=self.tolerance
        )

    def test_classes_on_array_input(self) -> None:
        """Testing instantiated class methods on scalar input"""
        _bundle_testing(
            bundle=self.testing_on_array_bundle, shape=(3,), tolerance=self.tolerance
        )

    def test_classes_on_array1_input(self) -> None:
        """Testing instantiated class methods on scalar input of size 1"""
        _bundle_testing(
            bundle=self.testing_on_array1_bundle, shape=(1,), tolerance=self.tolerance
        )


class OrbitAttitudeCurveWrapperTest(unittest.TestCase):
    """Testing orbit and attitutde boresight 3d curve wrapping"""

    def setUp(self):
        _state_vectors = np.array(
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

        _yaw = np.array([-10.9353126825047] * 160)
        _pitch = np.array([-19.9452555208155] * 160)
        _roll = np.array([-57.1252699331354] * 160)

        _time_axis = RegularAxis(
            (0, 5, _state_vectors.size // 3),
            PreciseDateTime.from_utc_string("17-FEB-2020 16:00:34.908999712209"),
        )
        _ypr_matrix = np.vstack((_yaw, _pitch, _roll))

        # creating orbit and orbit curve wrapper
        self.orbit = GeneralSarOrbit(_time_axis, _state_vectors)
        self.gso_curve = GSO3DCurveWrapper(orbit=self.orbit)

        # creating attitude and attitude curve wrapper
        self.attitude = GeneralSarAttitude(
            self.orbit, _time_axis, _ypr_matrix, "RPY", "ZERODOPPLER"
        )
        self.arf_boresight_normal_curve = (
            create_attitude_boresight_normal_curve_wrapper(attitude=self.attitude)
        )

        self.t_ref = PreciseDateTime.from_utc_string(
            "17-FEB-2020 16:06:59.908999712209"
        )

        gso_eval_ref = np.array([4400055.07025084, 764330.555188024, 4540503.56953453])
        gso_eval_der_ref = np.array(
            [-140.98703293372247, -24.4907610098969, 139.80664584670166]
        )
        gso_eval_der2_ref = np.array(
            [-0.00433043231922322, -0.0007522365727994565, -0.004490308497434352]
        )
        arf_eval_ref = np.array(
            [-0.9158216926726779, 0.021905849099771817, 0.40098723297053845]
        )
        arf_eval_der_ref = np.array(
            [-1.2408405794970916e-05, -2.1554559165450146e-06, -2.822202099198846e-05]
        )
        arf_eval_der2_ref = np.array(
            [8.769651232371575e-10, 1.5224460808385037e-10, -3.8726503663491334e-10]
        )

        # setup 3d curve testing on scalar input
        self.curve_testing_scalar_bundle = [
            TestingBundle(
                method="eval",
                curve=self.gso_curve,
                params=self.t_ref,
                expected_output=gso_eval_ref,
            ),
            TestingBundle(
                method="eval",
                curve=self.arf_boresight_normal_curve,
                params=self.t_ref,
                expected_output=arf_eval_ref,
            ),
            TestingBundle(
                method="eval_der1",
                curve=self.gso_curve,
                params=self.t_ref,
                expected_output=gso_eval_der_ref,
            ),
            TestingBundle(
                method="eval_der1",
                curve=self.arf_boresight_normal_curve,
                params=self.t_ref,
                expected_output=arf_eval_der_ref,
            ),
            TestingBundle(
                method="eval_der2",
                curve=self.gso_curve,
                params=self.t_ref,
                expected_output=gso_eval_der2_ref,
            ),
            TestingBundle(
                method="eval_der2",
                curve=self.arf_boresight_normal_curve,
                params=self.t_ref,
                expected_output=arf_eval_der2_ref,
            ),
        ]

        # setup 3d curve testing on array input
        self.curve_testing_array_bundle = [
            TestingBundle(
                method="eval",
                curve=self.gso_curve,
                params=np.array([self.t_ref] * 4),
                expected_output=np.broadcast_to(gso_eval_ref, (4, 3)),
            ),
            TestingBundle(
                method="eval",
                curve=self.arf_boresight_normal_curve,
                params=np.array([self.t_ref] * 4),
                expected_output=np.broadcast_to(arf_eval_ref, (4, 3)),
            ),
            TestingBundle(
                method="eval_der1",
                curve=self.gso_curve,
                params=np.array([self.t_ref] * 4),
                expected_output=np.broadcast_to(gso_eval_der_ref, (4, 3)),
            ),
            TestingBundle(
                method="eval_der1",
                curve=self.arf_boresight_normal_curve,
                params=np.array([self.t_ref] * 4),
                expected_output=np.broadcast_to(arf_eval_der_ref, (4, 3)),
            ),
            TestingBundle(
                method="eval_der2",
                curve=self.gso_curve,
                params=np.array([self.t_ref] * 4),
                expected_output=np.broadcast_to(gso_eval_der2_ref, (4, 3)),
            ),
            TestingBundle(
                method="eval_der2",
                curve=self.arf_boresight_normal_curve,
                params=np.array([self.t_ref] * 4),
                expected_output=np.broadcast_to(arf_eval_der2_ref, (4, 3)),
            ),
        ]

        # setup 3d curve testing on array input of size 1
        self.curve_testing_array1_bundle = [
            TestingBundle(
                method="eval",
                curve=self.gso_curve,
                params=np.array([self.t_ref]),
                expected_output=gso_eval_ref.reshape(1, 3),
            ),
            TestingBundle(
                method="eval",
                curve=self.arf_boresight_normal_curve,
                params=np.array([self.t_ref]),
                expected_output=arf_eval_ref.reshape(1, 3),
            ),
            TestingBundle(
                method="eval_der1",
                curve=self.gso_curve,
                params=np.array([self.t_ref]),
                expected_output=gso_eval_der_ref.reshape(1, 3),
            ),
            TestingBundle(
                method="eval_der1",
                curve=self.arf_boresight_normal_curve,
                params=np.array([self.t_ref]),
                expected_output=arf_eval_der_ref.reshape(1, 3),
            ),
            TestingBundle(
                method="eval_der2",
                curve=self.gso_curve,
                params=np.array([self.t_ref]),
                expected_output=gso_eval_der2_ref.reshape(1, 3),
            ),
            TestingBundle(
                method="eval_der2",
                curve=self.arf_boresight_normal_curve,
                params=np.array([self.t_ref]),
                expected_output=arf_eval_der2_ref.reshape(1, 3),
            ),
        ]

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

    def test_curve3d_scalar_input(self) -> None:
        """Testing curve methods on scalar input for each 3D curve object instantiated"""
        _bundle_testing_3d(
            bundle=self.curve_testing_scalar_bundle,
            shape=(3,),
            tolerance=self.tolerance,
        )

    def test_curve3d_array_input(self) -> None:
        """Testing curve methods on array input for each 3D curve object instantiated"""
        _bundle_testing_3d(
            bundle=self.curve_testing_array_bundle,
            shape=(4, 3),
            tolerance=self.tolerance,
        )

    def test_curve3d_array1_input(self) -> None:
        """Testing curve methods on array input of size 1 for each 3D curve object instantiated"""
        _bundle_testing_3d(
            bundle=self.curve_testing_array1_bundle,
            shape=(1, 3),
            tolerance=self.tolerance,
        )


if __name__ == "__main__":
    unittest.main()
