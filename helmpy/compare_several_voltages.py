
"""
HELMpy, open source package of power flow solvers developed on Python 3 
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np

from helmpy.root_path import ROOT_PATH


def main(
    *,
    print_all,
    base_profile_xlsx,
    comparison_xlsx_list,
):
    """
    Compare Several Voltages

    :param print_all:
    :param base_profile_xlsx:
    :param comparison_xlsx_list:
    :return:
    """
    pd.set_option('display.max_rows',1000)
    pd.set_option('display.max_columns',1000)
    pd.set_option('display.width',1000)

    ############   Modify these files names   ###################
    # base profile
    file_base = base_profile_xlsx
    file_list = comparison_xlsx_list

    data = pd.read_excel(file_base, sheet_name="Buses")
    magnitude_1 = data['Voltages Magnitude']
    phase_angles_1 = data["Voltages Phase Angle"]

    for file in file_list:
        data = pd.read_excel(file,sheet_name="Buses")
        magnitude_2 = data['Voltages Magnitude']
        phase_angles_2 = data["Voltages Phase Angle"]

        M = np.zeros(len(magnitude_1))
        P_A = np.zeros(len(magnitude_1))

        if(print_all):
            print("\n---------1---------")
            for i in range(len(magnitude_1)):
                print(i,magnitude_1[i],phase_angles_1[i])
            print("\n---------2---------")
            for i in range(len(magnitude_1)):
                print(i,magnitude_2[i],phase_angles_2[i])

        if(print_all):
            print("\n----Differences between 1 and 2-----")
        for i in range(len(magnitude_1)):
            M[i] = abs(magnitude_1[i]-magnitude_2[i])
            P_A[i] = abs(phase_angles_1[i]-phase_angles_2[i])
            if(print_all):
                print(i,"Magnitude:",M[i],"Phase Angles:",P_A[i])

        print("Differences between \"%s\" y \"%s\""%(file_base,file))
        print("Highest magnitude difference: ",np.max(M))
        print("Highest Phase Angles difference: ",np.max(P_A))
        print("\n\n")


if __name__ == '__main__':
    main(
        base_profile_xlsx= ROOT_PATH / 'data' / 'results' / 'Results HELM DS M2 PV2 case118 1.02 1e-08.xlsx',
        comparison_xlsx_list=[
            ROOT_PATH / 'data' / 'results' / 'Results HELM DS M1 PV1 case118 1.02 1e-08.xlsx',
            ROOT_PATH / 'data' / 'results' / 'Results HELM DS M1 PV2 case118 1.02 1e-08.xlsx',
        ],
        print_all=False
    )
