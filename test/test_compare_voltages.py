from compare_voltages import main
from root_path import ROOT_PATH


def test_compare_voltages():
    main(
        xlsx_0_file_path=ROOT_PATH /
                         'Results HELM DS M1 PV1 case118 1.02 1e-08.xlsx',
        xlsx_1_file_path=ROOT_PATH /
                         'Results HELM DS M1 PV2 case118 1.02 1e-08.xlsx',
    )
