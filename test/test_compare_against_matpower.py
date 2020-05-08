import parent_path
from helmpy.compare.compare_with_matpower import main
from helmpy.util.root_path import ROOT_PATH


def test_compare_against_matpower():
    matpower_csv_file_path = ROOT_PATH / 'data' / 'results' / 'Results MATPOWER case118 buses.csv'
    helm_csv_file_path = ROOT_PATH / 'data' / 'results' / 'Results HELM PV1 case118 1.02 1e-08 buses.csv'

    maximum_voltage_magnitude_difference, maximum_phase_angle_difference = \
        main(
            matpower_csv_file_path=matpower_csv_file_path,
            helm_csv_file_path=helm_csv_file_path,
            print_all=True,
        )

    assert maximum_voltage_magnitude_difference < 1e-2
    assert maximum_phase_angle_difference < 0.6


if __name__ == '__main__':
    test_compare_against_matpower()
