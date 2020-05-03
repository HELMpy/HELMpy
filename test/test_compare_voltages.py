import pytest

from helmpy.compare.compare_voltages import main
from helmpy.util.root_path import ROOT_PATH


@pytest.mark.parametrize(
    'csv_0_file_path, csv_1_file_path',
    [
        (
            ROOT_PATH / 'data' / 'results' /
            'Results HELM DS M1 PV1 case118 1.02 1e-08 buses.csv',
            ROOT_PATH / 'data' / 'results' /
            'Results HELM DS M1 PV2 case118 1.02 1e-08 buses.csv'
         ),
        (
            ROOT_PATH / 'data' / 'results' /
            'Results HELM DS M2 PV1 case118 1.02 1e-08 buses.csv',
            ROOT_PATH / 'data' / 'results' /
            'Results HELM DS M2 PV2 case118 1.02 1e-08 buses.csv'
        ),
        (
                ROOT_PATH / 'data' / 'results' /
                'Results HELM PV1 case118 1.02 1e-08 buses.csv',
                ROOT_PATH / 'data' / 'results' /
                'Results HELM PV2 case118 1.02 1e-08 buses.csv'
        ),
    ],
    ids=(
        'helm ds m1 pv1/pv2',
        'helm ds m2 pv1/pv2',
        'helm pv1/pv2',
    )
)
def test_compare_voltages(csv_0_file_path, csv_1_file_path):
    maximum_voltage_magnitude_difference, maximum_phase_angle_difference = \
        main(
            buses_csv_0_file_path=csv_0_file_path,
            buses_csv_1_file_path=csv_1_file_path,
        )

    assert maximum_voltage_magnitude_difference < 1e-12
    assert maximum_phase_angle_difference < 1e-12
