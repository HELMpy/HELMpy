"""
HELMpy, open source package of power flow solvers developed on Python 3 
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import cmath as cm
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized

from helmpy.core.helm_pade import Pade
from helmpy.core.helm_functions import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)


# Global variables that will be the basic parameters of the main function
detailed_run_print = False   #Print details on the run
Mis = 1e-4      #Mismatch
case = ''       #results file name
scale = 1       #load scale
N_coef = 100    #Max number of coefficients
Q_limits = True     #Checks Q generation limits

# Global variables declaration
N = 1
V = np.ones(N)
Pg = np.zeros(N)
Qg = np.zeros(N)
Pd = np.zeros(N)
Qd = np.zeros(N)
Pi = np.zeros(N)
Si = np.zeros(N, dtype=complex)
Qgmax = np.zeros(N)
Qgmin = np.zeros(N)
Buses_type = [0 for i in range(N)]
Shunt = np.zeros(N, dtype=complex)
Y = np.zeros((N,N), dtype=complex)
Yre = np.zeros((N,N), dtype=float)
Yimag = np.zeros((N,N), dtype=float)
Yshunt = np.zeros(N, dtype=complex)
Ploss = np.float64()
buses = 0
branches = 0
N_branches = 0
Ytrans = np.zeros((N,N), dtype=complex)
Pg_sch = np.copy(Pg)
N_generators = 0
generators = 0
Number_bus = dict()
slack_bus = 0
slack = 0
branches_buses = []
conduc_buses = []
phase_dict = dict()
phase_barras = []
V_complex_profile = np.zeros(N, dtype=complex)
first_check = True
pade_til = 0
list_gen = np.zeros(1, dtype=int)
num_gen = 0
list_gen_remove = []
list_coef = []
Flag_divergence = False
Power_branches = np.zeros((N_branches,8), dtype=float)
Ybr_list = list()
Power_print = pd.DataFrame()
Pmismatch = 0
S_gen = 0
S_load = 0
S_mismatch = 0


# Arrays and data lists creation
def initialize_data_arrays():
    global V, Pg, Qg, Pd, Qd, Buses_type, Qgmax, Qgmin, N, Y, Yshunt, Shunt, Ytrans
    global Pg_sch, Number_bus, N_branches
    global branches_buses, phase_barras, Pi, conduc_buses, V_complex_profile
    global N_generators, list_gen, Yre, Yimag, V_polar_final, list_gen_remove
    global list_coef, phase_dict, first_check, Power_branches, Ybr_list
    V = np.ones(N)
    Pg = np.zeros(N)
    Qg = np.zeros(N)
    Pd = np.zeros(N)
    Qd = np.zeros(N)
    Pi = np.zeros(N)

    Qgmax = np.zeros(N)
    Qgmin = np.zeros(N)
    Buses_type = [0 for i in range(N)]
    Shunt = np.zeros(N, dtype=complex)
    Y = np.zeros((N,N), dtype=complex)
    Yre = np.zeros((N,N), dtype=float)
    Yimag = np.zeros((N,N), dtype=float)
    Pg_sch = np.copy(Pg)
    Yshunt = np.zeros(N, dtype=complex)
    Ytrans = np.zeros((N,N), dtype=complex)
    Number_bus = dict()
    branches_buses = [[i] for i in range(N)]
    phase_dict = dict()
    phase_barras = np.full(N, False)
    conduc_buses = np.full(N, False)
    V_complex_profile = np.zeros(N, dtype=complex)
    list_gen = np.zeros(N_generators-1, dtype=int)
    V_polar_final = np.zeros((N,2), dtype=float)
    list_gen_remove = []
    first_check = True
    list_coef = []
    Ybr_list = list()
    Power_branches = np.zeros((N_branches,8), dtype=float)


# Modified Y matrix
Ytrans_mod = np.zeros((2*N+2,2*N+2),dtype=float)
# Column list of PV and PVLIM admittance. Also contains the PV buses: [PV bus i, non-modified admittance column]
Y_Vsp_PV = []
# Create modified Y matrix and list that contains the respective column to its voltage on PV and PVLIM buses 
def Modif_Ytrans():
    global Ytrans, Ytrans_mod,Y_Vsp_PV, N, Buses_type, K, slack, solve, branches_buses
    global num_gen, list_gen
    Ytrans_mod = np.zeros((2*N+2,2*N+2),dtype=float)
    Y_Vsp_PV =[]

    for i in range(N):
        if Buses_type[i]=='Slack':
            Ytrans_mod[2*i][2*i]=1
            Ytrans_mod[2*i + 1][2*i + 1]=1
        else:
            for j in branches_buses[i]:
                Ytrans_mod[2*i][2*j]=Ytrans[i][j].real
                Ytrans_mod[2*i][2*j + 1]=Ytrans[i][j].imag*-1
                Ytrans_mod[2*i + 1][2*j]=Ytrans[i][j].imag
                Ytrans_mod[2*i + 1][2*j + 1]=Ytrans[i][j].real

    # Penultimate row and last row
    for j in branches_buses[slack]:
        Ytrans_mod[2*N][2*j]=Ytrans[slack][j].real
        Ytrans_mod[2*N][2*j + 1]=Ytrans[slack][j].imag*-1
        Ytrans_mod[2*N + 1][2*j]=Ytrans[slack][j].imag
        Ytrans_mod[2*N + 1][2*j + 1]=Ytrans[slack][j].real

    # Penultimate columns
    for i in list_gen:
        Ytrans_mod[i*2][2*N]=-K[i]
    Ytrans_mod[2*N][2*N]=-K[slack]

    # Last column
    Ytrans_mod[2*N + 1][2*N + 1] = 1

    for i in list_gen:
        if( slack in branches_buses[i]):
            array = np.zeros( 2*len(branches_buses[i])+2, dtype=float)  
        else:  
            array = np.zeros( 2*len(branches_buses[i]), dtype=float)
        pos = 0
        for k in branches_buses[i]:
            array[pos] = Ytrans_mod[2*k][2*i]
            array[pos+1] = Ytrans_mod[2*k+1][2*i]
            Ytrans_mod[2*k][2*i] = 0
            Ytrans_mod[2*k+1][2*i] = 0
            pos += 2
        if( slack in branches_buses[i]):
            array[pos] = Ytrans_mod[2*N][2*i]
            array[pos+1] = Ytrans_mod[2*N+1][2*i]
            Ytrans_mod[2*N][2*i] = 0
            Ytrans_mod[2*N+1][2*i] = 0
        Y_Vsp_PV.append([i, array.copy()])
        Ytrans_mod[2*i + 1][2*i]=1

    # Return a function for solving a sparse linear system, with Ytrans_mod pre-factorized.
    solve = factorized(csc_matrix(Ytrans_mod))


# List of unknowns
unknowns = []
# Coefficients array
coefficients = np.zeros((2*N+2,N_coef), dtype=float)
# Evaluated solutions columns array
Soluc_eval = np.zeros((2*N+2,N_coef), dtype=float)
# Functions list to evaluate
Soluc_no_eval = []
# Arrays and lists creation
def Unknowns_soluc():
    global Buses_type, unknowns, N, N_coef, coefficients, Soluc_no_eval, Soluc_eval
    unknowns = []
    coefficients = np.zeros((2*N+2,N_coef), dtype=float)
    Soluc_eval = np.zeros((2*N+2,N_coef), dtype=float)
    Soluc_no_eval = []
    for i in range(N):
        if Buses_type[i]=='PV' or Buses_type[i]=='PVLIM':
            unknowns.append([i,'Q'])
            coefficients[2*i][0] = 0
            unknowns.append([i,'Vim'])
            coefficients[2*i + 1][0] = 0
            Soluc_no_eval.append([i,Gen_bus])
        else:
            unknowns.append([i,'Vre'])
            coefficients[2*i][0] = 1
            unknowns.append([i,'Vim'])
            coefficients[2*i + 1][0] = 0
            if Buses_type[i] == 'PQ':
                Soluc_no_eval.append([i, evaluate_pq_load_buses_equation])
            else:
                Soluc_no_eval.append([i, evaluate_slack_bus_equation])
    Soluc_no_eval.append([N,P_Loss_Q_slack])
    unknowns.append([N, 'PLoss'])
    coefficients[2*N][0] = 0
    unknowns.append([N+1,'Qslack'])
    coefficients[2*N+1][0] = 0


# Real voltage array for PV and PVLIM buses
Vre_PV = np.zeros((N, N_coef), dtype=float)
# Real voltage of PV and PVLIM buses computing
def Calculo_Vre_PV(n): # coefficient n
    global Vre_PV, V_complex, V, list_gen
    for i in list_gen:
        if n == 0:
            Vre_PV[i][n] = 1
        elif n == 1:
            Vre_PV[i][n] = (V[i]**2 - 1)/2
        else:
            aux = 0
            for k in range(1,n):
                aux += V_complex[i][k] * np.conj(V_complex[i][n-k])
            Vre_PV[i][n] = -aux/2

# Actualized complex voltages array of each bus (Vre+Vim sum)
V_complex = np.zeros((N, N_coef), dtype=complex)
# Complex voltages computing
def compute_complex_voltages(n):  # coefficient n
    global V_complex, Vre_PV, coefficients, Buses_type, N
    for i in range(N):
        if Buses_type[i]=='PV' or Buses_type[i]=='PVLIM':
            V_complex[i][n] = Vre_PV[i][n] + 1j*coefficients[i*2 + 1][n]
        else:
            V_complex[i][n] = coefficients[i*2][n] + 1j*coefficients[i*2 + 1][n]


# Inverse voltages "W" array
W = np.ones((N, N_coef), dtype=complex)
# W computing
def calculate_inverse_voltages_w_array(n):
    global W, V_complex, N
    if n > 0:
        for i in range(N):
            aux = 0
            for k in range(n):
                aux += (W[i][k] * V_complex[i][n-k])
            W[i][n] = -aux


# Function to evaluate the PV bus equation for the slack bus 
def P_Loss_Q_slack(nothing,n):  # coefficient n
    global Pg, Pd, Soluc_eval, W, Yshunt, V_complex, coefficients, phase_barras
    global phase_dict, Pi, K, slack, N
    i = slack
    aux = 0
    aux_Ploss = 0
    for k in range(1,n):
        aux += coefficients[2*N + 1][k]*np.conj(W[i][n-k])
        aux_Ploss += coefficients[N*2][k]*np.conj(W[i][n-k])
    if phase_barras[i]:
        PP = 0
        for k in range(len(phase_dict[i][0])):
            PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1]
        result = Pi[i]*np.conj(W[i][n-1]) - Yshunt[i]*V_complex[i][n-1]  - PP - aux*1j + K[i]*aux_Ploss
    else:
        result = Pi[i]*np.conj(W[i][n-1]) - Yshunt[i]*V_complex[i][n-1] - aux*1j + K[i]*aux_Ploss
    Soluc_eval[2*N + 1][n] = np.imag(result)
    Soluc_eval[2*N][n] = np.real(result)


# Function to evaluate the slack bus equation 
def evaluate_slack_bus_equation(i,n):  # bus i, coefficient n
    global V, Soluc_eval
    if n == 1:
        Soluc_eval[2*i][n] = V[i] - 1


# Function to evaluate the PQ buses equation 
def evaluate_pq_load_buses_equation(i,n):  # bus i, coefficient n
    global Pg, Pd, Qg, Qd, Si, Soluc_eval, W, Yshunt, V_complex, phase_barras
    global phase_dict
    if phase_barras[i]:
        PP = 0
        for k in range(len(phase_dict[i][0])):
            PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1]
        result = np.conj(Si[i])*np.conj(W[i][n-1]) - Yshunt[i]*V_complex[i][n-1] - PP
    else:
        result = np.conj(Si[i])*np.conj(W[i][n-1]) - Yshunt[i]*V_complex[i][n-1]
    Soluc_eval[2*i][n] = np.real(result)
    Soluc_eval[2*i + 1][n] = np.imag(result)


# Function to evaluate the PV buses equation            
def Gen_bus(i,n):  # bus i, coefficient n
    global Pg, Pd, Soluc_eval, W, Yshunt, V_complex, coefficients, phase_barras
    global phase_dict, Pi, K
    aux = 0
    aux_Ploss = 0
    for k in range(1,n):
        aux += coefficients[i*2][k]*np.conj(W[i][n-k])
        aux_Ploss += coefficients[N*2][k]*np.conj(W[i][n-k])
    if phase_barras[i]:
        PP = 0
        for k in range(len(phase_dict[i][0])):
            PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1]
        result = Pi[i]*np.conj(W[i][n-1]) - Yshunt[i]*V_complex[i][n-1]  - PP - aux*1j + K[i]*aux_Ploss
    else:
        result = Pi[i]*np.conj(W[i][n-1]) - Yshunt[i]*V_complex[i][n-1] - aux*1j + K[i]*aux_Ploss
    Soluc_eval[2*i][n] = np.real(result)
    Soluc_eval[2*i + 1][n] = np.imag(result)

# Vector of the participation factors: K's
K = np.zeros(N, dtype=float)


# Series coefficients counter (to mismatch) 
series_large = 0
# Loop of coefficients computing until the mismatch is reached
def computing_voltages_mismatch():
    global series_large, Y_Vsp_PV, Soluc_no_eval, Vre_PV, Soluc_eval, coefficients, N, Mis, Ytrans_mod, branches_buses
    global V_complex, W, Pi, Si, Pg, Pd, Qg, Qd, V_complex_profile, first_check, pade_til, solve, list_gen
    global list_coef, detailed_run_print, Flag_divergence, Q_limits
    global Buses_type, Vre, Vimag, Qgmax, Qgmin, Yre, Yimag, branches_buses, list_gen, list_gen_remove # Por funcion check limits 

    Vre_PV = np.zeros((N, N_coef), dtype=float)
    V_complex = np.zeros((N, N_coef), dtype=complex)
    W = np.ones((N, N_coef), dtype=complex)
    Flag_recalculate = 1
    Flag_divergence = False
    Calculo_Vre_PV(0)
    compute_complex_voltages(0)
    Pi = Pg - Pd
    Si = Pi + Qg*1j - Qd*1j
    coef_actual = 0
    series_large = 1
    while True:
        coef_actual += 1
        if detailed_run_print:
            print("Computing coefficient: %d"%coef_actual)
        Calculo_Vre_PV(coef_actual)
        for i in range(len(Soluc_no_eval)):
            Soluc_no_eval[i][1](Soluc_no_eval[i][0],coef_actual)
        resta_columnas_PV = np.zeros(2*N+2, dtype=float)
        for Vre_vec in Y_Vsp_PV:
            array = Vre_PV[Vre_vec[0]][coef_actual] * Vre_vec[1]
            pos = 0
            for k in branches_buses[Vre_vec[0]]:
                resta_columnas_PV[2*k] += array[pos]
                resta_columnas_PV[2*k+1] += array[pos+1]
                pos += 2
            if( slack in branches_buses[Vre_vec[0]]):
                resta_columnas_PV[2*N] += array[pos]
                resta_columnas_PV[2*N+1] += array[pos+1]
        aux = Soluc_eval[:,coef_actual] - resta_columnas_PV

        # New column of coefficients
        coefficients[:,coef_actual] = solve(aux)

        compute_complex_voltages(coef_actual)
        calculate_inverse_voltages_w_array(coef_actual)
        
        # Mismatch check
        Flag_Mismatch = 0
        series_large += 1
        if (series_large - 1) % 2 == 0 and series_large > 3:
            if first_check:
                first_check = False
                for i in range(N):
                    magn1,rad1 = cm.polar(Pade(V_complex[i],series_large-2))
                    V_complex_profile[i] = Pade(V_complex[i],series_large)
                    magn2, rad2 = cm.polar(V_complex_profile[i])
                    if((abs(magn1-magn2)>Mis) or (abs(rad1-rad2)>Mis)):
                        Flag_Mismatch = 1
                        pade_til = i+1
                        break
            else:
                seguir = True
                for i in range(pade_til):
                    magn1,rad1 = cm.polar(V_complex_profile[i])
                    V_complex_profile[i] = Pade(V_complex[i],series_large)
                    magn2, rad2 = cm.polar(V_complex_profile[i])
                    if((abs(magn1-magn2)>Mis) or (abs(rad1-rad2)>Mis)):
                        Flag_Mismatch = 1
                        pade_til = i+1
                        seguir = False
                        break
                if seguir:
                    for i in range(pade_til,N):
                        magn1,rad1 = cm.polar(Pade(V_complex[i],series_large-2))
                        V_complex_profile[i] = Pade(V_complex[i],series_large)
                        magn2, rad2 = cm.polar(V_complex_profile[i])
                        if((abs(magn1-magn2)>Mis) or (abs(rad1-rad2)>Mis)):
                            Flag_Mismatch = 1
                            pade_til = i+1
                            break
            if Flag_Mismatch == 0:
                # Qgen check or ignore limits
                Vre, Vimag = separate_complex_to_real_imag_voltages(V_complex_profile)
                if(Q_limits):
                    flag_violacion, Qg, Buses_type = Check_PVLIM(
                        Qg, Qd, Qgmax, Qgmin, Vre, Vimag, Yre, Yimag,
                        branches_buses, list_gen, list_gen_remove, Buses_type,
                        detailed_run_print,
                    )
                    if flag_violacion:
                        if detailed_run_print:
                            print("At coefficient %d the system is to be resolved due to PVLIM to PQ switches\n"%series_large)
                        list_coef.append(series_large)
                        Flag_recalculate = False
                        return Flag_recalculate
                print('\nConvergence has been reached. %d coefficients were calculated'%series_large)
                list_coef.append(series_large)
                break
        if series_large > N_coef-1:
            print('\nThe problem has no physical solution')
            Flag_divergence = 1
            break
    return Flag_recalculate


# Separate each voltage value in magnitude and phase angle (degrees). Calculate Ploss
V_polar_final = np.zeros((N,2), dtype=float)


# Main loop
def helm_ds_m1_pv1(
        grid_data_file_path,
        Print_Details=False, Mismatch=1e-4, Scale=1,
        MaxCoefficients=100, Enforce_Qlimits=True, DSB_model=True,
        Results_FileName='', Save_results=False,
):
    global V_complex_profile, N, buses, branches, N_branches, N_coef, N_generators, generators, T, Flag_divergence
    global detailed_run_print, Mis, case, scale, Q_limits
    global Power_print, V_polar_final, list_coef, S_gen, S_load, S_mismatch, Ploss, Pmismatch #Nuevos globales necesarias para la funcion write
    global Ybr_list, Shunt, slack, Pd, Qd, Pg, Qg, K, list_gen ##Nuevos globales necesarias para la funcion powerbalance
    global Vre, Vimag, Yre, Yimag , branches_buses ##Nuevos globales necesarias para la funcion  Q_in y separate Vre Vimag
    global V, Qgmax, Qgmin, slack_bus, Buses_type, Y, Yshunt, Ytrans, conduc_buses ##Nuevos globales necesarias para la funcion process_case_data
    global Pg_sch, num_gen, phase_barras, phase_dict ##Nuevos globales necesarias para la funcion process_case_data
    global list_gen_remove

    if (type(Print_Details) is not bool or \
        type(Mismatch) is not float or \
        type(Results_FileName)is not str or \
        not(
                type(Scale) is float or
                type(Scale) is int
        ) or \
        type(MaxCoefficients) is not int or \
        type(Enforce_Qlimits) is not bool or \
        type(DSB_model) is not bool
    ):
        print("Erroneous argument type.")
        return

    algorithm = 'HELM DS M1 PV1'
    detailed_run_print = Print_Details
    Mis = Mismatch
    if(Results_FileName==''):
        case = grid_data_file_path[0:-5]
    else:
        case = Results_FileName
    scale = Scale
    N_coef = MaxCoefficients
    Q_limits = Enforce_Qlimits

    buses = pd.read_excel(grid_data_file_path, sheet_name='Buses', header=None)
    branches = pd.read_excel(grid_data_file_path, sheet_name='Branches', header=None)
    generators = pd.read_excel(grid_data_file_path, sheet_name='Generators', header=None)

    N = len(buses.index)
    N_generators = len(generators.index)
    N_branches = len(branches.index)

    initialize_data_arrays()

    (   Buses_type, V, Qgmax, Qgmin, Pd, Qd, Pg, Shunt, buses, branches, N_branches,
        slack_bus, N, N_generators, generators, Number_bus, Yshunt, Ytrans, Y,
        conduc_buses, slack, list_gen, num_gen, Yre, Yimag, scale,
        branches_buses, phase_barras, phase_dict, Ybr_list, Pg_sch,
    ) = \
    preprocess_case_data(
        algorithm, scale, 
        buses, N,
        branches, N_branches,
        generators, N_generators,
        N_coef,
    )

    while True:
        # Re-construct list_gen. List of generators (PV buses)
        list_gen = np.setdiff1d(list_gen, list_gen_remove, assume_unique=True)
        # Computing of the K factor for each PV bus and the slack bus.
        K, Pg = compute_k_factor(Pg, Pg_sch, Pd, slack, N, list_gen)
        # Set the slack's participation factor to 1 and the rest to 0. Classic slack bus model.
        if not(DSB_model):
            K_slack_1(K,slack)
        # Create modified Y matrix and list that contains the respective column to its voltage on PV and PVLIM buses 
        Modif_Ytrans()
        # Arrays and lists creation
        Unknowns_soluc()
        # Loop of coefficients computing until the mismatch is reached
        if computing_voltages_mismatch():
            break
    if not Flag_divergence:
        if detailed_run_print or Save_results:
            Ploss = Pade(coefficients[2*N],series_large)
            (Power_branches, S_gen, S_load, S_mismatch, Pmismatch) = power_balance(
                V_complex_profile, Ybr_list,
                N_branches, N, Shunt, slack, Pd, Qd, Pg, Qg,
                S_gen, S_load, S_mismatch, Q_limits, list_gen,
                Vre, Vimag, Yre, Yimag , branches_buses, algorithm,
                K=K, Pmismatch=Pmismatch
            )
            V_polar_final = convert_complex_to_polar_voltages(V_complex_profile, N)
            if detailed_run_print:
                print_voltage_profile(V_polar_final,N)
                print(create_power_balance_string(
                    scale, Mis, algorithm,
                    list_coef, S_gen, S_load, S_mismatch,
                    Ploss, Pmismatch
                ))
            if Save_results:
                write_results_on_files(
                case, scale, Mis, algorithm,
                V_polar_final, V_complex_profile, Power_branches,
                list_coef, S_gen, S_load, S_mismatch,
                Ploss, Pmismatch
            )
        return V_complex_profile
