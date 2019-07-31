from helmpy.compare.compare_several_voltages import main
from helmpy.util.root_path import ROOT_PATH


def test_compare_several_voltages():
    base_profile_csv = ROOT_PATH / 'data' / 'results' / 'Results HELM DS M1 PV1 case118 1.02 1e-08 buses.csv'
    comparison_csv_list = [
        ROOT_PATH / 'data' / 'results' /
        'Results HELM DS M1 PV2 case118 1.02 1e-08 buses.csv',
        ROOT_PATH / 'data' / 'results' /
        'Results HELM DS M2 PV1 case118 1.02 1e-08 buses.csv',
        ROOT_PATH / 'data' / 'results' /
        'Results HELM DS M2 PV2 case118 1.02 1e-08 buses.csv',
    ]
    maximum_voltage_magnitude_difference, maximum_phase_angle_difference = \
        main(
            base_profile_csv=base_profile_csv,
            comparison_csv_list=comparison_csv_list,
            print_all=False
        )

    assert maximum_voltage_magnitude_difference < 1e-12
    assert maximum_phase_angle_difference < 1e-12
