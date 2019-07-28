
"""
HELMpy, open source package of power flow solvers developed on Python 3 
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np


def main(
        *,
        xlsx_0_file_path,
        xlsx_1_file_path,
):
    """
    Compare Voltages

    :param xlsx_0_file_path:
    :param xlsx_1_file_path:
    :return:
    """
    pd.set_option('display.max_rows',1000)
    pd.set_option('display.max_columns',1000)
    pd.set_option('display.width',1000)

    ############   Modify these files names   ###################
    data1 = pd.read_excel(
        xlsx_0_file_path,
        sheet_name="Buses",
    )
    data2 = pd.read_excel(
        xlsx_1_file_path,
        sheet_name="Buses",
    )

    magnitude_1 = data1['Voltages Magnitude']
    phase_angles_1 = data1["Voltages Phase Angle"]
    magnitude_2 = data2['Voltages Magnitude']
    phase_angles_2 = data2["Voltages Phase Angle"]

    print_all = False ###############################

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

    print("Highest magnitude difference: ",np.max(M))
    print("Highest Phase Angles difference: ",np.max(P_A))


if __name__ == '__main__':
    main(
        xlsx_0_file_path='Results HELM DS M1 PV1 case118 1.02 1e-08.xlsx',
        xlsx_1_file_path='Results HELM DS M1 PV2 case118 1.02 1e-08.xlsx',
    )
