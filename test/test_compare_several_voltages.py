from helmpy.compare.compare_several_voltages import main
from helmpy.util.root_path import ROOT_PATH


def test_compare_several_voltages():
    maximum_voltage_magnitude_difference, maximum_phase_angle_difference = \
        main(
            base_profile_xlsx= ROOT_PATH / 'data' / 'results' / 'Results HELM DS M1 PV1 case118 1.02 1e-08.xlsx',
            comparison_xlsx_list=[
                ROOT_PATH / 'data' / 'results' / 'Results HELM DS M1 PV2 case118 1.02 1e-08.xlsx',
                ROOT_PATH / 'data' / 'results' / 'Results HELM DS M2 PV1 case118 1.02 1e-08.xlsx',
                ROOT_PATH / 'data' / 'results' / 'Results HELM DS M2 PV2 case118 1.02 1e-08.xlsx',
            ],
            print_all=False
        )

    assert maximum_voltage_magnitude_difference < 1e-12
    assert maximum_phase_angle_difference < 1e-12
