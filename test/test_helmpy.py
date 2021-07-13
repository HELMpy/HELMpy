"""
Test helmpy
"""

from os.path import basename 

import numpy as np
import pandas as pd

from paths import helmpy, HELMPY_PATH



class CaseDataAndResults:
    def __init__(
        self,
        grid_data_file_path,
        results_classic_slack_file_path,
        results_distributed_slack_file_path,
    ):
        #  Name
        self.name = basename(grid_data_file_path[0:-5])
        # case object
        self.case = helmpy.create_case_data_object_from_xlsx(grid_data_file_path)
        self.grid_data_file_path = grid_data_file_path
        # Classic slack
        case_data = pd.read_excel(results_classic_slack_file_path, sheet_name="Buses")
        self.classic_slack_magnitude = np.array(case_data['Voltages Magnitude'])
        self.classic_slack_phase_angles = np.array(case_data["Voltages Phase Angle"])
        # Distributed slack
        case_data = pd.read_excel(results_distributed_slack_file_path, sheet_name="Buses")
        self.distributed_slack_magnitude = np.array(case_data['Voltages Magnitude'])
        self.distributed_slack_phase_angles = np.array(case_data["Voltages Phase Angle"])


def convert_complex_to_polar_voltages(complex_voltage):
    """Separate each voltage value in magnitude and phase angle (degrees)"""
    polar_voltage = np.zeros((len(complex_voltage),2), dtype=float)
    polar_voltage[:,0] = np.absolute(complex_voltage)
    polar_voltage[:,1] = np.angle(complex_voltage, deg=True)
    return polar_voltage


def algorithm_str(pv_bus_model, DSB_model, DSB_model_method):
    """Construct algorithm string"""
    pv_bus_model_str = 'PV1' if pv_bus_model == 1 else 'PV2'
    if DSB_model:
        DSB_model_method_str = 'M1' if DSB_model_method == 1 else 'M2'
        algorithm = 'HELM DS ' + DSB_model_method_str + ' ' + pv_bus_model_str
    elif DSB_model_method is not None: # DSB_model = False
        DSB_model_method_str = 'M1' if DSB_model_method == 1 else 'M2'
        algorithm = 'HELM DS ' + DSB_model_method_str + ' ' + pv_bus_model_str + ' DSB_model=False'
    else:
        algorithm = 'HELM ' + pv_bus_model_str
    return algorithm


def test_helmpy_functions(detailed_print, cases_to_test, pv_dsb_methods): 
    """
    Test every pv_dsb_methods and case.
    """
    # Add all the error here for a final and fast check
    total_errors = []
    print("\n  ###############################")
    print("  ### Check Check Check Check ###")
    print("  ###############################")
    for pv_bus_model, DSB_model, DSB_model_method in pv_dsb_methods:
        if detailed_print:
            print("\n##########   " + algorithm_str(pv_bus_model, DSB_model, DSB_model_method) + '   ##########')
        for case in cases_to_test:
            if detailed_print:
                print("\nCase: " + case.name)
            scale = 1.02 if DSB_model else 1
            # Execute function
            complex_voltage = helmpy.helm(
                case.case, mismatch=1e-8, scale=scale, # detailed_run_print=True,
                pv_bus_model=pv_bus_model, DSB_model=DSB_model, DSB_model_method=DSB_model_method )
            # Errors
            polar_voltage = convert_complex_to_polar_voltages( complex_voltage ) # Calculate polar voltage
            if DSB_model:
                magnitud_error = np.absolute( polar_voltage[:,0] - case.distributed_slack_magnitude )
                phase_angles_error = np.absolute( polar_voltage[:,1] - case.distributed_slack_phase_angles )
            else:
                magnitud_error = np.absolute( polar_voltage[:,0] - case.classic_slack_magnitude )
                phase_angles_error = np.absolute( polar_voltage[:,1] - case.classic_slack_phase_angles )
            max_magnitud_error = np.max(magnitud_error)
            max_phase_angles_error = np.max(phase_angles_error)
            if detailed_print:
                print("Maximun magnitud error: ", max_magnitud_error)
                print("Maximun phase angles error: ", max_phase_angles_error)
            total_errors.append(max_magnitud_error)
            total_errors.append(max_phase_angles_error)
    
    # Final output
    print("\n\n  ###################################")
    print("  ### Maximun error of all tests: ###")
    print("  ###################################\n")
    print("--->", np.max(total_errors), end='\n\n')

    return total_errors


if __name__ == '__main__':

    # Uncomment every pv_model/dsb_method and case that wants to be tested.

    # Variable to extend print
    detailed_print = True

    # pv_models/dsb_methods to test
    pv_dsb_methods = [ # pv_bus_model, DSB_model, DSB_model_method
        (1, False, None),     # HELM PV1
        (2, False, None),     # HELM PV2
        (1, False, 1),        # HELM DS M1 PV1 DSB_model=False
        (2, False, 1),        # HELM DS M1 PV2 DSB_model=False
        (1, True, 1),         # HELM DS M1 PV1
        (2, True, 1),         # HELM DS M1 PV2
        (1, False, 2),        # HELM DS M2 PV1 DSB_model=False
        (2, False, 2),        # HELM DS M2 PV2 DSB_model=False
        (1, True, 2),         # HELM DS M2 PV1
        (2, True, 2),         # HELM DS M2 PV2
    ]

    # Define the object of every test case

    case9 = CaseDataAndResults(
        str( HELMPY_PATH / 'data' / 'cases' / "case9.xlsx" ),
        str( HELMPY_PATH / 'data' / 'results' / "Results HELM case9 1 1e-08.xlsx" ),
        str( HELMPY_PATH / 'data' / 'results' / "Results HELM DS case9 1.02 1e-08.xlsx" ),
    )

    case118 = CaseDataAndResults(
        str( HELMPY_PATH / 'data' / 'cases' / "case118.xlsx" ),
        str( HELMPY_PATH / 'data' / 'results' / "Results HELM case118 1 1e-08.xlsx" ),
        str( HELMPY_PATH / 'data' / 'results' / "Results HELM DS case118 1.02 1e-08.xlsx" ),
    )

    case1354pegase = CaseDataAndResults(
        str( HELMPY_PATH / 'data' / 'cases' / "case1354pegase.xlsx" ),
        str( HELMPY_PATH / 'data' / 'results' / "Results HELM case1354pegase 1 1e-08.xlsx" ),
        str( HELMPY_PATH / 'data' / 'results' / "Results HELM DS case1354pegase 1.02 1e-08.xlsx" ),
    )

    case2869pegase = CaseDataAndResults(
        str( HELMPY_PATH / 'data' / 'cases' / "case2869pegase.xlsx" ),
        str( HELMPY_PATH / 'data' / 'results' / "Results HELM case2869pegase 1 1e-08.xlsx" ),
        str( HELMPY_PATH / 'data' / 'results' / "Results HELM DS case2869pegase 1.02 1e-08.xlsx" ),
    )

    # Cases to test
    cases_to_test = [
            case9,
            case118,
            case1354pegase,
            case2869pegase,
    ]

    test_helmpy_functions(detailed_print, cases_to_test, pv_dsb_methods)
