from compare_with_matpower import main
from root_path import ROOT_PATH


def test_compare_with_matpower():
    main(
        # MATPOWER file name
        matpower_xlsx_file_path=ROOT_PATH / 'Results MATPOWER case118.xlsx',
        # HELMpy file name
        helm_xlsx_file_path=ROOT_PATH / 'Results HELM PV1 case118 1.02 1e-08.xlsx',
        print_all=True,
    )


if __name__ == '__main__':
    test_compare_with_matpower()
