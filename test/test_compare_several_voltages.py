from compare_several_voltages import main
from root_path import ROOT_PATH


def test_compare_several_voltages():
    main(
        base_profile_xlsx= ROOT_PATH / 'data' / 'results' / 'Results HELM DS M2 PV2 case118 1.02 1e-08.xlsx',
        comparison_xlsx_list=[
            ROOT_PATH / 'data' / 'results' / 'Results HELM DS M1 PV1 case118 1.02 1e-08.xlsx',
            ROOT_PATH / 'data' / 'results' / 'Results HELM DS M1 PV2 case118 1.02 1e-08.xlsx',
        ],
        print_all=False
    )
