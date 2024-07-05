# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/orbit.py Orbit object"""

import unittest

import numpy as np

from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve
from arepytools.geometry.geometric_functions import (
    compute_ground_velocity_from_trajectory,
    compute_incidence_angles_from_trajectory,
    compute_look_angles_from_trajectory,
)
from arepytools.geometry.orbit import ExtrapolationNotAllowed, Orbit
from arepytools.io.create_orbit import create_orbit
from arepytools.io.metadata import StateVectors
from arepytools.timing.precisedatetime import PreciseDateTime

pos_x = [
    -2542286.449576481,
    -2542066.5079547316,
    -2541845.6347068036,
    -2541623.8297894364,
    -2541401.0931600337,
    -2541177.424776237,
    -2540952.8245959855,
    -2540727.2925774334,
    -2540500.828679304,
    -2540273.432860546,
    -2540045.1050804476,
    -2539815.8452985254,
    -2539585.653474864,
    -2539354.5295697646,
    -2539122.47354382,
    -2538889.4853578685,
    -2538655.5649734233,
    -2538420.712352204,
    -2538184.9274560884,
    -2537948.21024718,
]
pos_y = [
    -5094859.4894666,
    -5092594.238796566,
    -5090327.468904538,
    -5088059.180544005,
    -5085789.374468956,
    -5083518.051433686,
    -5081245.212193193,
    -5078970.857502782,
    -5076694.988118281,
    -5074417.604795827,
    -5072138.708292266,
    -5069858.299364754,
    -5067576.378770969,
    -5065292.947268894,
    -5063008.0056172125,
    -5060721.554574907,
    -5058433.594901471,
    -5056144.127356704,
    -5053853.152701135,
    -5051560.671695597,
]
pos_z = [
    3901083.7183820857,
    3904175.2612526673,
    3907265.6186505724,
    3910354.7896390846,
    3913442.7732819766,
    3916529.5686432645,
    3919615.1747874315,
    3922699.5907792132,
    3925782.8156838706,
    3928864.8485669065,
    3931945.688494257,
    3935025.334532101,
    3938103.785747145,
    3941181.0412063445,
    3944257.099977135,
    3947331.9611271904,
    3950405.6237246613,
    3953478.086837955,
    3956549.3495360035,
    3959619.410887985,
]
vel_x = [
    439,
    440,
    442,
    444,
    446,
    448,
    450,
    452,
    454,
    455,
    457,
    459,
    461,
    463,
    465,
    466,
    468,
    469,
    472,
    474,
]
vel_y = [
    4529,
    4532,
    4535,
    4538,
    4544,
    4547,
    4550,
    4552,
    4554,
    4555,
    4557,
    4560,
    4565,
    4568,
    4572,
    4574,
    4578,
    4580,
    4583,
    4586,
]
vel_z = [
    6184,
    6181,
    6179,
    6177,
    6174,
    6172,
    6170,
    6167,
    6165,
    6162,
    6160,
    6158,
    6155,
    6153,
    6150,
    6148,
    6146,
    6144,
    6141,
    6138,
]
DT = 0.5
time_axis_relative = np.arange(0, 10, DT)
time_axis_origin = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:56.000000")


class OrbitProtocolComplianceTest(unittest.TestCase):
    """Testing Orbit compliance with TwiceDifferentiable3DCurve protocol"""

    def test_orbit_protocol_compliance(self) -> None:
        """Testing Orbit protocol compliance"""
        self.assertIsInstance(Orbit, TwiceDifferentiable3DCurve)


class OrbitTest(unittest.TestCase):
    """Testing Orbit generation, properties and methods"""

    def setUp(self) -> None:
        self._state_vectors_metadata = StateVectors(
            position_vector=np.stack([pos_x, pos_y, pos_z], axis=1).ravel().tolist(),
            velocity_vector=np.stack([vel_x, vel_y, vel_z], axis=1).ravel().tolist(),
            t_ref_utc=time_axis_origin,
            dt_sv_s=DT,
        )
        # expected results
        self._tolerance = 1e-6
        self._times = (
            np.array([0.67, 2.56, 3.23, 5.8, 7.3, 8.4, 9.28]) + time_axis_origin
        )
        self._expected_pos = np.array(
            [
                [-2541991.51667295, -5091823.70786736, 3905226.12193514],
                [-2541150.5219575, -5083245.3907668, 3916899.90420767],
                [-2540849.19560279, -5080199.19720824, 3921034.15400884],
                [-2539677.84205497, -5068489.32836387, 3936872.54869185],
                [-2538982.79233349, -5061636.31586544, 3946102.16084407],
                [-2538467.75565909, -5056602.13916507, 3952863.69473292],
                [-2538052.45005212, -5052569.50979786, 3958268.80885579],
            ]
        )
        self._expected_vel = np.array(
            [
                [441.4473803, 4533.05342227, 6181.09986721],
                [448.49214067, 4544.52639022, 6172.11662186],
                [450.98948273, 4548.58823598, 6168.92732246],
                [460.57005549, 4564.14349903, 6156.66343748],
                [466.16286972, 4573.20402512, 6149.48291368],
                [470.25419444, 4579.82631009, 6144.18013695],
                [473.58261429, 4585.18911946, 6140.06974106],
            ]
        )
        self._expected_acc = np.array(
            [
                [3.77763037, 6.095279, -5.02893538],
                [3.72759827, 6.06460755, -4.75951289],
                [3.72737768, 6.06038945, -4.76204088],
                [3.72838663, 6.04500478, -4.78121927],
                [3.72227767, 6.02750557, -4.80892479],
                [3.59289969, 5.85534651, -5.14276888],
                [2.60597199, 4.59067105, -7.62766475],
            ]
        )

    def test_orbit_creation(self) -> None:
        """Test Orbit creation through create_orbit() function"""
        orbit = create_orbit(state_vectors=self._state_vectors_metadata)
        self.assertIsInstance(orbit, Orbit)

    def test_orbit_properties(self) -> None:
        """Test Orbit properties"""
        orbit = create_orbit(state_vectors=self._state_vectors_metadata)
        np.testing.assert_array_equal(
            orbit.positions, np.stack([pos_x, pos_y, pos_z], axis=1)
        )
        np.testing.assert_array_equal(
            orbit.velocities, np.stack([vel_x, vel_y, vel_z], axis=1)
        )
        delta_times = orbit.times - (time_axis_relative + time_axis_origin)
        np.testing.assert_array_equal(
            delta_times.astype(float), np.zeros_like(delta_times, dtype=float)
        )

    def test_orbit_methods(self) -> None:
        """Test Orbit evaluate methods"""
        orbit = create_orbit(state_vectors=self._state_vectors_metadata)
        np.testing.assert_allclose(
            orbit.evaluate(self._times),
            self._expected_pos,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            orbit.evaluate_first_derivatives(self._times),
            self._expected_vel,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            orbit.evaluate_second_derivatives(self._times),
            self._expected_acc,
            atol=self._tolerance,
            rtol=0,
        )

    def test_orbit_methods_extrapolation_error_1(self) -> None:
        """Test Orbit evaluate methods raising extrapolation error"""
        orbit = create_orbit(state_vectors=self._state_vectors_metadata)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate(self._times[0] - 2)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate_first_derivatives(self._times[-1] + 12)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate_second_derivatives(self._times[0] - 2)

    def test_orbit_methods_extrapolation_error_2(self) -> None:
        """Test Orbit evaluate methods raising extrapolation error"""
        orbit = create_orbit(state_vectors=self._state_vectors_metadata)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate(self._times - 200)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate_first_derivatives(self._times + 200)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate_second_derivatives(self._times - 200)

    def test_orbit_methods_extrapolation_error_3(self) -> None:
        """Test Orbit evaluate methods raising extrapolation error"""
        orbit = create_orbit(state_vectors=self._state_vectors_metadata)
        test_times = self._times.copy()
        test_times[3] = self._times[0] + 18
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate(test_times)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate_first_derivatives(test_times)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate_second_derivatives(test_times)

    def test_orbit_methods_extrapolation_error_4(self) -> None:
        """Test Orbit evaluate methods raising extrapolation error"""
        orbit = create_orbit(state_vectors=self._state_vectors_metadata)
        test_times = self._times.copy()
        test_times[2] = self._times[0] - 9
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate(test_times)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate_first_derivatives(test_times)
        with self.assertRaises(ExtrapolationNotAllowed):
            orbit.evaluate_second_derivatives(test_times)


class AnglesComputationFromOrbitTest(unittest.TestCase):
    """Testing angles computation from Orbit object"""

    def setUp(self) -> None:
        self._state_vectors_metadata = StateVectors(
            position_vector=np.stack([pos_x, pos_y, pos_z], axis=1).ravel().tolist(),
            velocity_vector=np.stack([vel_x, vel_y, vel_z], axis=1).ravel().tolist(),
            t_ref_utc=time_axis_origin,
            dt_sv_s=DT,
        )
        self._trajectory = create_orbit(state_vectors=self._state_vectors_metadata)
        self._range_times = np.array(
            [0.00362255, 0.003623, 0.0036239, 0.003635, 0.003639, 0.003642, 0.003645]
        )
        self._azimuth_time = time_axis_origin + 2
        self._geocoding_side = "RIGHT"
        # expected results
        self._tolerance = 1e-9
        self._expected_look_angles = np.array(
            [
                0.20348233735250404,
                0.2040352825385317,
                0.20513633655495583,
                0.21822227400375235,
                0.22273297530481562,
                0.22605128004217567,
                0.22931680025662243,
            ]
        )
        self._expected_incidence_angles = np.array(
            [
                0.22003649511651838,
                0.2206377522894163,
                0.22183504499141393,
                0.23606867691985423,
                0.2409767142368985,
                0.24458790669056735,
                0.24814215077561505,
            ]
        )

    def test_compute_look_angles_single_range(self) -> None:
        """Testing compute_look_angles_from_trajectory with a single range value"""
        look_angles = compute_look_angles_from_trajectory(
            trajectory=self._trajectory,
            azimuth_time=self._azimuth_time,
            range_times=self._range_times[0],
            look_direction=self._geocoding_side,
        )
        np.testing.assert_allclose(
            look_angles, self._expected_look_angles[0], atol=self._tolerance, rtol=0
        )

    def test_compute_look_angles_range_array(self) -> None:
        """Testing compute_look_angles_from_trajectory with a range array"""
        look_angles = compute_look_angles_from_trajectory(
            trajectory=self._trajectory,
            azimuth_time=self._azimuth_time,
            range_times=self._range_times,
            look_direction=self._geocoding_side,
        )
        np.testing.assert_allclose(
            look_angles, self._expected_look_angles, atol=self._tolerance, rtol=0
        )

    def test_compute_incidence_angles_single_range(self) -> None:
        """Testing compute_incidence_angles_from_trajectory with a single range value"""
        incidence_angles = compute_incidence_angles_from_trajectory(
            trajectory=self._trajectory,
            azimuth_time=self._azimuth_time,
            range_times=self._range_times[0],
            look_direction=self._geocoding_side,
        )
        np.testing.assert_allclose(
            incidence_angles,
            self._expected_incidence_angles[0],
            atol=self._tolerance,
            rtol=0,
        )

    def test_compute_incidence_angles_range_array(self) -> None:
        """Testing compute_incidence_angles_from_trajectory with a range array"""
        incidence_angles = compute_incidence_angles_from_trajectory(
            trajectory=self._trajectory,
            azimuth_time=self._azimuth_time,
            range_times=self._range_times,
            look_direction=self._geocoding_side,
        )
        np.testing.assert_allclose(
            incidence_angles,
            self._expected_incidence_angles,
            atol=self._tolerance,
            rtol=0,
        )


class GroundVelocityFromOrbitTest(unittest.TestCase):
    """Testing ground velocity computation from Orbit object"""

    def setUp(self) -> None:
        self._state_vectors_metadata = StateVectors(
            position_vector=np.stack([pos_x, pos_y, pos_z], axis=1).ravel().tolist(),
            velocity_vector=np.stack([vel_x, vel_y, vel_z], axis=1).ravel().tolist(),
            t_ref_utc=time_axis_origin,
            dt_sv_s=DT,
        )
        self._az_time = time_axis_origin + 2.3
        self._trajectory = create_orbit(state_vectors=self._state_vectors_metadata)
        self._look_angles = np.deg2rad(np.arange(15.0, 50.0, 5.0))
        # expected results
        self._tolerance = 1e-6
        self._expected_velocities = np.array(
            [
                7073.866866931723,
                7068.0743880794025,
                7061.324649385491,
                7053.284329192286,
                7043.447653796795,
                7031.009082103516,
                7014.600593133645,
            ]
        )

    def test_compute_ground_velocity_from_trajectory_0(self) -> None:
        """Test compute_ground_velocity_from_trajectory, case 0"""
        ground_velocities = compute_ground_velocity_from_trajectory(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_angles_rad=self._look_angles[0],
        )
        self.assertIsInstance(ground_velocities, float)
        np.testing.assert_allclose(
            ground_velocities,
            self._expected_velocities[0],
            atol=self._tolerance,
            rtol=0,
        )

    def test_compute_ground_velocity_from_trajectory_1(self) -> None:
        """Test compute_ground_velocity_from_trajectory, case 0"""
        ground_velocities = compute_ground_velocity_from_trajectory(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_angles_rad=self._look_angles,
        )
        np.testing.assert_allclose(
            ground_velocities, self._expected_velocities, atol=self._tolerance, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
