import unittest

import numpy as np

from arepytools.io import metadata
from arepytools.io.parsing.metadata_parsing import parse_metadata, serialize_metadata
from arepytools.io.parsing.translate import translate_metadata_to_model
from arepytools.timing.precisedatetime import PreciseDateTime

METADATA = """<?xml version="1.0" encoding="utf-8"?>
<AresysXmlDoc>
  <NumberOfChannels>1</NumberOfChannels>
  <VersionNumber>2.1</VersionNumber>
  <Description/>
  <Channel Number="1" Total="1">
    <RasterInfo>
      <FileName>GRD_0001</FileName>
      <Lines>8951</Lines>
      <Samples>2215</Samples>
      <HeaderOffsetBytes>150</HeaderOffsetBytes>
      <RowPrefixBytes>20</RowPrefixBytes>
      <ByteOrder>LITTLEENDIAN</ByteOrder>
      <CellType>FLOAT32</CellType>
      <LinesStep unit="s">0.00325489654798287</LinesStep>
      <SamplesStep unit="m">25.0</SamplesStep>
      <LinesStart unit="Utc">11-JAN-2017 05:06:05.420354672133</LinesStart>
      <SamplesStart unit="m">0.0</SamplesStart>
    </RasterInfo>
    <DataSetInfo>
      <SensorName>BIOMASS</SensorName>
      <Description>NOT_AVAILABLE - AZIMUTH FOCUSED RANGE COMPENSATED</Description>
      <SenseDate>NOT_AVAILABLE</SenseDate>
      <AcquisitionMode>STRIPMAP</AcquisitionMode>
      <ImageType>MULTILOOK</ImageType>
      <Projection>GROUND RANGE</Projection>
      <AcquisitionStation>NOT_AVAILABLE</AcquisitionStation>
      <ProcessingCenter>NOT_AVAILABLE</ProcessingCenter>
      <ProcessingDate>10-FEB-2023 18:59:39.063638000000</ProcessingDate>
      <ProcessingSoftware>GSS</ProcessingSoftware>
      <fc_hz>435000000.0</fc_hz>
      <SideLooking>LEFT</SideLooking>
    </DataSetInfo>
    <SwathInfo>
      <Swath>S2</Swath>
      <SwathAcquisitionOrder>9</SwathAcquisitionOrder>
      <Polarization>H/H</Polarization>
      <Rank>8</Rank>
      <RangeDelayBias unit="s">0.05</RangeDelayBias>
      <AcquisitionStartTime unit="Utc">11-JAN-2017 05:06:05.420354672133</AcquisitionStartTime>
      <AzimuthSteeringRateReferenceTime unit="s">-2.3</AzimuthSteeringRateReferenceTime>
      <AzimuthSteeringRatePol>
        <val N="1">45.0</val>
        <val N="2">42.0</val>
        <val N="3">43.0</val>
      </AzimuthSteeringRatePol>
      <AcquisitionPRF>20.0</AcquisitionPRF>
      <EchoesPerBurst>30</EchoesPerBurst>
    </SwathInfo>
    <SamplingConstants>
      <frg_hz unit="Hz">10.0</frg_hz>
      <Brg_hz unit="Hz">20.0</Brg_hz>
      <faz_hz unit="Hz">30.0</faz_hz>
      <Baz_hz unit="Hz">40.0</Baz_hz>
    </SamplingConstants>
    <DataStatistics>
      <NumSamples>19826458</NumSamples>
      <MaxI>2676823552.0</MaxI>
      <MinI>83823.4921875</MinI>
      <MaxQ>0.0</MaxQ>
      <MinQ>0.0</MinQ>
      <SumI>89122400845249.0</SumI>
      <SumQ>0.0</SumQ>
      <Sum2I>7.00840507541525E20</Sum2I>
      <Sum2Q>0.0</Sum2Q>
      <StdDevI>3891349.90059871</StdDevI>
      <StdDevQ>0.0</StdDevQ>
    </DataStatistics>
    <StateVectorData>
      <OrbitNumber>NOT_AVAILABLE</OrbitNumber>
      <Track>7</Track>
      <OrbitDirection>ASCENDING</OrbitDirection>
      <pSV_m>
        <val N="1">5317607.32991368</val>
        <val N="2">610604.181576807</val>
        <val N="3">4577936.42495716</val>
        <val N="4">5313024.92248479</val>
        <val N="5">608285.759542598</val>
        <val N="6">4583546.68380492</val>
      </pSV_m>
      <vSV_mOs>
        <val N="1">-4579.23071779759</val>
        <val N="2">-2318.41093794023</val>
        <val N="3">5612.88065438087</val>
        <val N="4">-4585.59543140451</val>
        <val N="5">-2318.43362350978</val>
        <val N="6">5607.64864044348</val>
      </vSV_mOs>
      <t_ref_Utc>11-JAN-2017 05:05:55.374776000000</t_ref_Utc>
      <dtSV_s unit="s">0.99999999962462</dtSV_s>
      <nSV_n>2</nSV_n>
    </StateVectorData>
    <SlantToGround Number="1" Total="1">
      <pol>
        <val N="1" unit="m">0.0279764859084441</val>
        <val N="2" unit="m/s">332392355.238117</val>
        <val N="3" unit="m/s">0</val>
        <val N="4" unit="m/s2">0</val>
        <val N="5" unit="m/s2">-147100378880.014</val>
        <val N="6" unit="m/s3">148771445722789</val>
        <val N="7" unit="m/s4">-1.18791374851972E17</val>
      </pol>
      <trg0_s unit="s">0.00498119194829745</trg0_s>
      <taz0_Utc unit="Utc">11-JAN-2017 05:06:19.987644172630</taz0_Utc>
    </SlantToGround>
    <GroundToSlant Number="1" Total="1">
      <pol>
        <val N="1" unit="s">0.00498119195039857</val>
        <val N="2" unit="s/m">3.00835380687427E-09</val>
        <val N="3" unit="s/m">0</val>
        <val N="4" unit="s/m2">0</val>
        <val N="5" unit="s/m2">4.02534653514271E-15</val>
        <val N="6" unit="s/m3">-2.45577629676765E-21</val>
        <val N="7" unit="s/m4">1.03753668287902E-28</val>
      </pol>
      <trg0_s unit="s">0.0</trg0_s>
      <taz0_Utc unit="Utc">11-JAN-2017 05:06:05.420354672133</taz0_Utc>
    </GroundToSlant>
    <AttitudeInfo>
      <t_ref_Utc>11-JAN-2017 05:05:55.374776000000</t_ref_Utc>
      <dtYPR_s>0.99999999962462</dtYPR_s>
      <nYPR_n>3</nYPR_n>
      <yaw_deg>
        <val N="1">6.01545966886107E-06</val>
        <val N="2">1.72217492445588E-05</val>
        <val N="3">2.15186524627847E-05</val>
      </yaw_deg>
      <pitch_deg>
        <val N="1">2.76923549635297E-06</val>
        <val N="2">7.98225683167924E-06</val>
        <val N="3">9.98095451338142E-06</val>
      </pitch_deg>
      <roll_deg>
        <val N="1">25.9269230465147</val>
        <val N="2">25.9265262380572</val>
        <val N="3">25.9261295216853</val>
      </roll_deg>
      <referenceFrame>ZERODOPPLER</referenceFrame>
      <rotationOrder>YPR</rotationOrder>
      <AttitudeType>NOMINAL</AttitudeType>
    </AttitudeInfo>
    <Pulse>
      <Direction>UP</Direction>
      <PulseLength unit="s">-1</PulseLength>
      <Bandwidth unit="Hz">-1</Bandwidth>
      <PulseEnergy unit="j">-1</PulseEnergy>
      <PulseSamplingRate unit="Hz">-1</PulseSamplingRate>
      <PulseStartFrequency unit="Hz">0.5</PulseStartFrequency>
      <PulseStartPhase unit="rad">0.785398163397448</PulseStartPhase>
    </Pulse>
  </Channel>
</AresysXmlDoc>
"""


class MetadataParsingSerializationTest(unittest.TestCase):
    def assertEqualMetadata(
        self, metadata_a: metadata.MetaData, metadata_b: metadata.MetaData
    ):
        model_a = translate_metadata_to_model(metadata_a)
        model_b = translate_metadata_to_model(metadata_b)
        self.assertEqual(model_a, model_b)

    def setUp(self) -> None:
        self.maxDiff = None

        self.metadata_str = METADATA
        self.metadata_obj = metadata.MetaData(description="")

        channel = metadata.MetaDataChannel()
        channel.number = 1
        channel.total = 1

        raster_info = metadata.RasterInfo(
            lines=8951,
            samples=2215,
            celltype="FLOAT32",
            filename="GRD_0001",
            header_offset_bytes=150,
            row_prefix_bytes=20,
            byteorder="LITTLEENDIAN",
            invalid_value=None,
            format_type=None,
        )
        raster_info.set_lines_axis(
            PreciseDateTime.from_utc_string("11-JAN-2017 05:06:05.420354672133"),
            "Utc",
            0.00325489654798287,
            "s",
        )
        raster_info.set_samples_axis(0.0, "m", 25.0, "m")
        channel.insert_element(raster_info)

        dataset_info = metadata.DataSetInfo(
            acquisition_mode_i="STRIPMAP", fc_hz_i=435000000.0
        )
        dataset_info.sensor_name = "BIOMASS"
        dataset_info.description = "NOT_AVAILABLE - AZIMUTH FOCUSED RANGE COMPENSATED"
        dataset_info.sense_date = None
        dataset_info.image_type = "MULTILOOK"
        dataset_info.projection = "GROUND RANGE"
        dataset_info.acquisition_station = "NOT_AVAILABLE"
        dataset_info.processing_center = "NOT_AVAILABLE"
        dataset_info.processing_date = PreciseDateTime.from_utc_string(
            "10-FEB-2023 18:59:39.063638000000"
        )
        dataset_info.processing_software = "GSS"
        dataset_info.side_looking = "LEFT"
        channel.insert_element(dataset_info)

        swath_info = metadata.SwathInfo(
            swath_i="S2", polarization_i="H/H", acquisition_prf_i=20.0
        )
        swath_info.acquisition_start_time = PreciseDateTime.from_utc_string(
            "11-JAN-2017 05:06:05.420354672133"
        )
        swath_info.azimuth_steering_rate_reference_time = -2.3
        swath_info.range_delay_bias = 0.05
        swath_info.echoes_per_burst = 30
        swath_info.rank = 8
        swath_info.swath_acquisition_order = 9
        swath_info.azimuth_steering_rate_pol = (45.0, 42.0, 43.0)
        channel.insert_element(swath_info)

        sampling_constants = metadata.SamplingConstants(10.0, 20.0, 30.0, 40.0)
        channel.insert_element(sampling_constants)

        data_statistics = metadata.DataStatistics(
            i_num_samples=19826458,
            i_max_i=2676823552.0,
            i_max_q=0.0,
            i_min_i=83823.4921875,
            i_min_q=0.0,
            i_sum_i=89122400845249.0,
            i_sum_q=0.0,
            i_sum_2_i=7.00840507541525e20,
            i_sum_2_q=0.0,
            i_std_dev_i=3891349.90059871,
            i_std_dev_q=0.0,
        )
        channel.insert_element(data_statistics)

        state_vectors = metadata.StateVectors(
            position_vector=np.array(
                [
                    5317607.32991368,
                    610604.181576807,
                    4577936.42495716,
                    5313024.92248479,
                    608285.759542598,
                    4583546.68380492,
                ]
            ).reshape((-1, 3)),
            velocity_vector=np.array(
                [
                    -4579.23071779759,
                    -2318.41093794023,
                    5612.88065438087,
                    -4585.59543140451,
                    -2318.43362350978,
                    5607.64864044348,
                ]
            ).reshape((-1, 3)),
            dt_sv_s=0.99999999962462,
            t_ref_utc=PreciseDateTime.from_utc_string(
                "11-JAN-2017 05:05:55.374776000000"
            ),
        )
        state_vectors.track_number = 7
        channel.insert_element(state_vectors)

        slant_to_ground = metadata.SlantToGroundVector(
            [
                metadata.SlantToGround(
                    i_ref_az=PreciseDateTime.from_utc_string(
                        "11-JAN-2017 05:06:19.987644172630"
                    ),
                    i_ref_rg=0.00498119194829745,
                    i_coefficients=[
                        0.0279764859084441,
                        332392355.238117,
                        0,
                        0,
                        -147100378880.014,
                        148771445722789,
                        -1.18791374851972e17,
                    ],
                )
            ]
        )
        channel.insert_element(slant_to_ground)

        ground_to_slant = metadata.GroundToSlantVector(
            [
                metadata.GroundToSlant(
                    i_ref_az=PreciseDateTime.from_utc_string(
                        "11-JAN-2017 05:06:05.420354672133"
                    ),
                    i_ref_rg=0.0,
                    i_coefficients=[
                        0.00498119195039857,
                        3.00835380687427e-09,
                        0,
                        0,
                        4.02534653514271e-15,
                        -2.45577629676765e-21,
                        1.03753668287902e-28,
                    ],
                )
            ]
        )
        channel.insert_element(ground_to_slant)

        attitude_info = metadata.AttitudeInfo(
            yaw=[
                6.01545966886107e-06,
                1.72217492445588e-05,
                2.15186524627847e-05,
            ],
            pitch=[
                2.76923549635297e-06,
                7.98225683167924e-06,
                9.98095451338142e-06,
            ],
            roll=[
                25.9269230465147,
                25.9265262380572,
                25.9261295216853,
            ],
            t0=PreciseDateTime.from_utc_string("11-JAN-2017 05:05:55.374776000000"),
            delta_t=0.99999999962462,
            ref_frame="ZERODOPPLER",
            rot_order="YPR",
        )
        attitude_info.attitude_type = "NOMINAL"
        channel.insert_element(attitude_info)

        pulse = metadata.Pulse(
            i_pulse_length=-1,
            i_bandwidth=-1,
            i_pulse_sampling_rate=-1,
            i_pulse_energy=-1,
            i_pulse_start_frequency=0.5,
            i_pulse_start_phase=0.785398163397448,
            i_pulse_direction="UP",
        )
        channel.insert_element(pulse)

        self.metadata_obj.append_channel(channel)

    def test_serialize(self):
        self.assertEqual(serialize_metadata(self.metadata_obj), self.metadata_str)

    def test_parse(self):
        self.assertEqualMetadata(parse_metadata(self.metadata_str), self.metadata_obj)


if __name__ == "__main__":
    unittest.main()
