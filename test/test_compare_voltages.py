from helmpy.compare_voltages import main
from helmpy.root_path import ROOT_PATH


def test_compare_voltages():
    xlsx_0_file_path = ROOT_PATH / 'data' / 'results' / \
                       'Results HELM DS M1 PV1 case118 1.02 1e-08.xlsx'
    xlsx_1_file_path = ROOT_PATH / 'data' / 'results' / \
                       'Results HELM DS M1 PV2 case118 1.02 1e-08.xlsx'
    main(
        xlsx_0_file_path=xlsx_0_file_path,
        xlsx_1_file_path=xlsx_1_file_path,
    )
