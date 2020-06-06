"""
HELMpy, open source package of power flow solvers developed on Python 3 
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import cmath as cm
import warnings
from os.path import basename 

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized

from helmpy.core.helm_pade import Pade
#from helmpy.core.helm_modified_functions import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)



# Arrays and data lists creation
def initialize_data_arrays(N):
    # Necesito inicializarlos adentro de las funciones (en alguna parte apropiada)
    Qg = np.zeros(N)
    V_complex_profile= np.zeros(N, dtype=complex)
    Vre = np.real(V_complex_profile)
    Vimag = np.imag(V_complex_profile)
    list_gen_remove = []
    list_coef = []
    return Qg, V_complex_profile, Vre, Vimag, list_gen_remove, list_coef


# Branches data processing to construct Ytrans, Yshunt, branches_buses and others
def porcess_branches(branches, N_branches, Number_bus, Yshunt, N):
    
    # Arrays and data lists creation
    Ytrans = np.zeros((N,N), dtype=complex)
    branches_buses = [[i] for i in range(N)]
    Ybr_list = list()
    phase_barras = np.full(N, False)
    phase_dict = dict()

    for i in range(N_branches):
        FromBus = branches[0][i]
        ToBus = branches[1][i]
        R = branches[2][i]
        X = branches[3][i]
        BTotal = branches[4][i]
        Tap = branches[8][i]
        Shift_degree = branches[9][i]

        FB = Number_bus[FromBus] 
        TB = Number_bus[ToBus]
        Ybr_list.append([FB, TB, np.zeros((2,2),dtype=complex)])
        Z = R + 1j*X
        if Tap == 0 or Tap == 1:
            if Z != 0:
                Yseries_ft = 1/Z
                if(Shift_degree==0):
                    Ybr_list[i][2][0,1] = Ybr_list[i][2][1,0] = -Yseries_ft
                else:
                    Shift = np.deg2rad(Shift_degree)
                    Yseries_ft_shift = Yseries_ft/(np.exp(-1j*Shift))
                    Yseries_tf_shift = Yseries_ft/(np.exp(1j*Shift))
                    Ybr_list[i][2][0,1] = -Yseries_ft_shift
                    Ybr_list[i][2][1,0] = -Yseries_tf_shift
                    if(phase_barras[FB]):
                        if( TB in phase_dict[FB][0]):
                            phase_dict[FB][1][ phase_dict[FB][0].index(TB) ] += Yseries_ft - Yseries_ft_shift
                        else:
                            phase_dict[FB][0].append(TB)
                            phase_dict[FB][1].append(Yseries_ft - Yseries_ft_shift)
                    else:
                        phase_dict[FB] = [[TB],[Yseries_ft - Yseries_ft_shift]]
                        phase_barras[FB] = True
                    if(phase_barras[TB]):
                        if( FB in phase_dict[TB][0]):
                            phase_dict[TB][1][ phase_dict[TB][0].index(FB) ] += Yseries_ft - Yseries_tf_shift
                        else:
                            phase_dict[FB][0].append(FB)
                            phase_dict[FB][1].append(Yseries_ft - Yseries_tf_shift)
                    else:
                        phase_dict[TB] = [[FB],[Yseries_ft - Yseries_tf_shift]]
                        phase_barras[TB] = True
                Ytrans[FB][TB] += -Yseries_ft
                Ytrans[FB][FB] +=  Yseries_ft
                Ytrans[TB][FB] += -Yseries_ft
                Ytrans[TB][TB] +=  Yseries_ft
            else:
                Ybr_list[i][2][0,1] = Ybr_list[i][2][1,0] = Yseries_ft = 0

            Bshunt_ft = 1j*BTotal/2
            Ybr_list[i][2][0,0] = Ybr_list[i][2][1,1] = Bshunt_ft + Yseries_ft
            Yshunt[FB] +=  Bshunt_ft
            Yshunt[TB] +=  Bshunt_ft
        else:
            Tap_inv = 1/Tap
            if Z != 0:
                Yseries_no_tap = 1/Z
                Yseries_ft = Yseries_no_tap * Tap_inv
                if(Shift_degree==0):
                    Ybr_list[i][2][0,1] = Ybr_list[i][2][1,0] = -Yseries_ft
                else:
                    Shift = np.deg2rad(Shift_degree)                
                    Yseries_ft_shift = Yseries_ft/(np.exp(-1j*Shift))
                    Yseries_tf_shift = Yseries_ft/(np.exp(1j*Shift))
                    Ybr_list[i][2][0,1] = -Yseries_ft_shift
                    Ybr_list[i][2][1,0] = -Yseries_tf_shift
                    if(phase_barras[FB]):
                        if( TB in phase_dict[FB][0]):
                            phase_dict[FB][1][ phase_dict[FB][0].index(TB) ] += Yseries_ft - Yseries_ft_shift
                        else:
                            phase_dict[FB][0].append(TB)
                            phase_dict[FB][1].append(Yseries_ft - Yseries_ft_shift)
                    else:
                        phase_dict[FB] = [[TB],[Yseries_ft - Yseries_ft_shift]]
                        phase_barras[FB] = True
                    if(phase_barras[TB]):
                        if( FB in phase_dict[TB][0]):
                            phase_dict[TB][1][ phase_dict[TB][0].index(FB) ] += Yseries_ft - Yseries_tf_shift
                        else:
                            phase_dict[FB][0].append(FB)
                            phase_dict[FB][1].append(Yseries_ft - Yseries_tf_shift)
                    else:
                        phase_dict[TB] = [[FB],[Yseries_ft - Yseries_tf_shift]]
                        phase_barras[TB] = True
                Ytrans[FB][TB] += -Yseries_ft
                Ytrans[FB][FB] +=  Yseries_ft
                Ytrans[TB][FB] += -Yseries_ft
                Ytrans[TB][TB] +=  Yseries_ft 
            else:
                Ybr_list[i][2][0,1] = Ybr_list[i][2][1,0] = Yseries_no_tap = Yseries_ft = 0
            
            B = 1j*BTotal/2
            Bshunt_f = (Yseries_no_tap + B)*(Tap_inv*Tap_inv) 
            Bshunt_t = Yseries_no_tap + B
            Ybr_list[i][2][0,0] = Bshunt_f
            Ybr_list[i][2][1,1] = Bshunt_t
            Yshunt[FB] +=  Bshunt_f - Yseries_ft
            Yshunt[TB] +=  Bshunt_t - Yseries_ft

        if TB not in branches_buses[FB]:
            branches_buses[FB].append(TB)
        if FB not in branches_buses[TB]:
            branches_buses[TB].append(FB)

    return(
        Ytrans,
        branches_buses, Ybr_list,
        phase_barras, phase_dict,
    )


# Processing of .xlsx file data
def preprocess_case_data(
    algorithm, scale, 
    buses, N,
    branches, N_branches,
    generators, N_generators,
    N_coef,
):
    Number_bus = dict()
    Buses_type = [0 for i in range(N)]
    conduc_buses = np.full(N, False)
    Yshunt = np.zeros(N, dtype=complex)
    V = np.ones(N)
    Pg = np.zeros(N)
    Qgmax = np.zeros(N)
    Qgmin = np.zeros(N)
    list_gen = np.zeros(N_generators-1, dtype=int)
    barras_CC = dict()

    Pd = buses[2]/100*scale
    Qd = buses[3]/100*scale
    Shunt = buses[5]*1j/100 + buses[4]/100

    for i in range(N):
        Number_bus[buses[0][i]] = i
        if(buses[1][i]!=3):
            Buses_type[i] = 'PQ'
        else:
            slack_bus = buses[0][i]
            slack = i
        Yshunt[i] =  Shunt[i]

    num_gen = N_generators-1
    pos = 0
    for i in range(N_generators):
        bus_i = Number_bus[generators[0][i]]
        if(bus_i!=slack):
            list_gen[pos] = bus_i
            pos += 1
        Buses_type[bus_i] = 'PVLIM'
        V[bus_i] = generators[5][i]
        Pg[bus_i] = generators[1][i]/100*scale
        Qgmax[bus_i] = generators[3][i]/100
        Qgmin[bus_i] = generators[4][i]/100
       
    Buses_type[slack] = 'Slack'
    Pg[slack] = 0
    if 'DS' in algorithm:
        Pg_sch = np.copy(Pg)

    (   Ytrans,
        branches_buses, Ybr_list,
        phase_barras, phase_dict,
    ) = \
    porcess_branches(branches, N_branches, Number_bus, Yshunt, N) 

    for i in range(N):
        branches_buses[i].sort()    # Variable that saves the branches
        if 'PV2' in algorithm:
            barras_CC[i] = np.zeros(N_coef, dtype=complex)    # Variable that saves the calculated values of PV buses

    Y = Ytrans.copy()
    for i in range(N):
        if( Yshunt[i].real != 0 ):
            conduc_buses[i] = True
        Y[i,i] += Yshunt[i]
        if phase_barras[i]:
            for k in range(len(phase_dict[i][0])):
                Y[i,phase_dict[i][0][k]] += phase_dict[i][1][k]
    Yre = np.real(Y)
    Yimag = np.imag(Y)

    if 'DS' in algorithm:
        return(
            Buses_type, V, Qgmax, Qgmin, Pd, Qd, Pg, Shunt, buses, branches, N_branches,
            slack_bus, N, N_generators, generators, Number_bus, Yshunt, Ytrans, Y,
            conduc_buses, slack, list_gen, num_gen, Yre, Yimag, scale,
            branches_buses, phase_barras, phase_dict, Ybr_list, Pg_sch
        )
    else:
        return(
            Buses_type, V, Qgmax, Qgmin, Pd, Qd, Pg, Shunt, buses, branches, N_branches,
            slack_bus, N, N_generators, generators, Number_bus, Yshunt, Ytrans, Y,
            conduc_buses, slack, list_gen, num_gen, Yre, Yimag, scale,
            branches_buses, phase_barras, phase_dict, Ybr_list
        )


# Create modified Y matrix and list that contains the respective column to its voltage on PV and PVLIM buses 
def Modif_Ytrans(
    Ytrans, N, Buses_type, K, slack, branches_buses, list_gen
):
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

    return solve, Y_Vsp_PV


# Arrays and lists creation
def Unknowns_soluc(Buses_type, N, N_coef):
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

    return unknowns, coefficients, Soluc_no_eval, Soluc_eval


# Real voltage of PV and PVLIM buses computing
def Calculo_Vre_PV(n, Vre_PV, V_complex, V, list_gen): # coefficient n
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


# W computing - Inverse voltages "W" array
def calculate_inverse_voltages_w_array(n, W, V_complex, N):
    for i in range(N):
        aux = 0
        for k in range(n):
            aux += (W[i][k] * V_complex[i][n-k])
        W[i][n] = -aux


# Function to evaluate the PV bus equation for the slack bus 
def P_Loss_Q_slack( # coefficient n
    i, n,
    Soluc_eval, N, V, Si, Pi, K, slack, 
    W, Yshunt, V_complex, coefficients,
    phase_barras, phase_dict,
):   
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
def evaluate_slack_bus_equation( # bus i, coefficient n
    i, n,
    Soluc_eval, N, V, Si, Pi, K, slack, 
    W, Yshunt, V_complex, coefficients,
    phase_barras, phase_dict,
):
    if n == 1:
        Soluc_eval[2*i][n] = V[i] - 1


# Function to evaluate the PQ buses equation 
def evaluate_pq_load_buses_equation( # bus i, coefficient n
    i, n,
    Soluc_eval, N, V, Si, Pi, K, slack, 
    W, Yshunt, V_complex, coefficients,
    phase_barras, phase_dict,
):
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
def Gen_bus( # bus i, coefficient n
    i, n,
    Soluc_eval, N, V, Si, Pi, K, slack, 
    W, Yshunt, V_complex, coefficients,
    phase_barras, phase_dict,
):
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


# Complex voltages computing
def compute_complex_voltages(n, V_complex, Vre_PV, coefficients, Buses_type, N):  # coefficient n
    for i in range(N):
        if Buses_type[i]=='PV' or Buses_type[i]=='PVLIM':
            V_complex[i][n] = Vre_PV[i][n] + 1j*coefficients[i*2 + 1][n]
        else:
            V_complex[i][n] = coefficients[i*2][n] + 1j*coefficients[i*2 + 1][n]


# Computing P injection at bus i. Must be used after Voltages_profile()
def P_iny(i, Vre, Vimag, Yre, Yimag, branches_buses):
    Piny = 0
    for k in branches_buses[i]:
        Piny += Vre[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) + Vimag[i]*(Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    return Piny


# Computing Q injection at bus i. Must be used after Voltages_profile()
def Q_iny(i, Vre, Vimag, Yre, Yimag, branches_buses):
    Qiny = 0
    for k in branches_buses[i]:
        Qiny += Vimag[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) - Vre[i]*(Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    return Qiny


# Create variable V_polar_final from V and tita_degree in NR
def create_polar_voltages_variable(V, tita_degree, N):
    polar_voltage = np.zeros((N,2), dtype=float)
    polar_voltage[:,0] = V  # Voltage magnitude
    polar_voltage[:,1] = tita_degree  # Voltage phase angle
    return polar_voltage


# Separate each voltage value in magnitude and phase angle (degrees)
def convert_complex_to_polar_voltages(complex_voltage, N):
    polar_voltage = np.zeros((N,2), dtype=float)
    polar_voltage[:,0] = np.absolute(complex_voltage)
    polar_voltage[:,1] = np.angle(complex_voltage, deg=True)
    return polar_voltage


# Verification of Qgen limits for PVLIM buses
def Check_PVLIM(
    Qg, Qd, Qgmax, Qgmin, Vre, Vimag, Yre, Yimag,
    branches_buses, list_gen, list_gen_remove, Buses_type,
    detailed_run_print,
):
    flag_violacion = False
    for i in list_gen:
        Qg_incog = Q_iny(i, Vre, Vimag, Yre, Yimag , branches_buses) + Qd[i]
        Qg[i] = Qg_incog
        if Qg_incog > Qgmax[i] or Qg_incog < Qgmin[i]:
            flag_violacion = True
            Buses_type[i] = 'PQ'
            list_gen_remove.append(i)
            Qg[i] = Qgmax[i] if Qg_incog > Qgmax[i] else Qgmin[i]
            if detailed_run_print:
                print('Bus %d exceeded its Qgen limit with %f. The exceeded limit %f will be assigned to the bus'%(i+1,Qg_incog,Qg[i]))
    return flag_violacion


# Computing of the K factor for each PV bus and the slack bus.
# Only the PV buses are considered to calculate Pgen_total. The PV buses that were converted to PQ buses are NOT considered.
def compute_k_factor(Pg, Pd, slack, N, list_gen):
    K = np.zeros(N, dtype=float)
    
    Pg[slack] = 0
    Pgen_total = 0
    Distrib = []
    # Active power that the slack must generate to compensate the system
    Pg[slack] = np.sum(Pd) - np.sum(Pg)
    for i in list_gen:
        if Pg[i] > 0:
            Pgen_total += Pg[i]
            Distrib.append(i)
    if Pg[slack] > 0:
        Pgen_total += Pg[slack]
        Distrib.append(slack)
    for i in Distrib:
        K[i] = Pg[i]/Pgen_total
        
    return K


# Set the slack's participation factor to 1 and the rest to 0. Classic slack bus model.
def K_slack_1(K,slack):
    K.fill(0)
    K[slack] = 1


# Loop of coefficients computing until the mismatch is reached
def computing_voltages_mismatch(
    Y_Vsp_PV, Soluc_no_eval, Soluc_eval, coefficients, N, Mis, 
    Pg, Pd, Qg, Qd, V_complex_profile, solve,
    list_coef, detailed_run_print, N_coef,
    Q_limits, Buses_type, Qgmax, Qgmin, Yre, Yimag, Vre, Vimag,
    branches_buses, list_gen_remove, list_gen, V,
    K, slack, Yshunt, phase_barras, phase_dict, # Function Gen_bus y otras
):
    Vre_PV = np.zeros((N, N_coef), dtype=float)
    V_complex = np.zeros((N, N_coef), dtype=complex)
    W = np.ones((N, N_coef), dtype=complex)
    Flag_recalculate = True
    Flag_divergence = False
    Calculo_Vre_PV(0, Vre_PV, V_complex, V, list_gen)
    compute_complex_voltages(0, V_complex, Vre_PV, coefficients, Buses_type, N)
    Pi = Pg - Pd
    Si = Pi + Qg*1j - Qd*1j
    coef_actual = 0
    series_large = 1
    first_check = True

    while True:
        coef_actual += 1
        if detailed_run_print:
            print("Computing coefficient: %d"%coef_actual)
        Calculo_Vre_PV(coef_actual, Vre_PV, V_complex, V, list_gen)
        for i in range(len(Soluc_no_eval)):
            Soluc_no_eval[i][1](
                Soluc_no_eval[i][0], coef_actual,
                Soluc_eval, N, V, Si, Pi, K, slack, 
                W, Yshunt, V_complex, coefficients,
                phase_barras, phase_dict,
            )
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

        compute_complex_voltages(coef_actual, V_complex, Vre_PV, coefficients, Buses_type, N)
        calculate_inverse_voltages_w_array(coef_actual, W, V_complex, N)
        
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
                if Q_limits:
                    flag_violacion = Check_PVLIM(
                        Qg, Qd, Qgmax, Qgmin, Vre, Vimag, Yre, Yimag,
                        branches_buses, list_gen, list_gen_remove, Buses_type,
                        detailed_run_print,
                    )
                    if flag_violacion:
                        if detailed_run_print:
                            print("At coefficient %d the system is to be resolved due to PVLIM to PQ switches\n"%series_large)
                        list_coef.append(series_large)
                        Flag_recalculate = False
                        return Flag_recalculate, Flag_divergence, series_large
                print('\nConvergence has been reached. %d coefficients were calculated'%series_large)
                list_coef.append(series_large)
                break
        if series_large > N_coef-1:
            print('\nThe problem has no physical solution')
            Flag_divergence = 1
            break
    return Flag_recalculate, Flag_divergence, series_large


# Computation of power flow trough branches and power balance
def power_balance(
    V_complex_profile, Ybr_list,
    N_branches, N, Shunt, slack, Pd, Qd, Pg, Qg,
    Q_limits, list_gen,
    Vre, Vimag, Yre, Yimag, branches_buses, algorithm,
    Pi=None, Qi=None, K=None
):
    Power_branches = np.zeros((N_branches,8), dtype=float)

    for branch in range(N_branches):

        Bus_from =  Power_branches[branch][0] = int(Ybr_list[branch][0])
        Bus_to = Power_branches[branch][1] = int(Ybr_list[branch][1])
        Ybr = Ybr_list[branch][2]

        V_from = V_complex_profile[Bus_from]
        V_to = V_complex_profile[Bus_to]
        V_vector = np.array([V_from,V_to])
        
        I =  np.matmul(Ybr,V_vector)

        S_ft = V_from * np.conj(I[0]) * 100
        S_tf = V_to * np.conj(I[1]) * 100
        S_branch_elements = S_ft + S_tf

        Power_branches[branch][2] = np.real(S_ft)
        Power_branches[branch][3] = np.imag(S_ft)

        Power_branches[branch][4] = np.real(S_tf)
        Power_branches[branch][5] = np.imag(S_tf)

        Power_branches[branch][6] = np.real(S_branch_elements)
        Power_branches[branch][7] = np.imag(S_branch_elements)

    P_losses_line = np.sum(Power_branches[:,6])/100
    Q_losses_line = np.sum(Power_branches[:,7]) * 1j /100

    # Computation of power through shunt capacitors, reactors or conductantes, Power balanca
    S_shunt = 0
    for i in range(N):
        if Shunt[i] != 0:
            S_shunt += V_complex_profile[i] * np.conj(V_complex_profile[i]*Shunt[i])

    Qload = np.sum(Qd) * 1j
    Pload = np.sum(Pd)

    if 'HELM' in algorithm:

        if not Q_limits:
            for i in list_gen:
                Qg[i] = Q_iny(i, Vre, Vimag, Yre, Yimag , branches_buses) + Qd[i]
        Qgen = (np.sum(Qg) + Q_iny(slack, Vre, Vimag, Yre, Yimag , branches_buses) + Qd[slack]) * 1j

        if 'DS' in algorithm: # algorithm models the distributed slack
            Pmismatch = P_losses_line + np.real(S_shunt)
            Pgen = np.sum(Pg + K*Pmismatch)
        else:
            Pgen = np.sum(Pg) + P_iny(slack, Vre, Vimag, Yre, Yimag, branches_buses) + Pd[slack]

    elif 'NR' in algorithm:

        if not Q_limits:
            for i in list_gen:
                Qg[i] = Qi[i] + Qd[i]
        Qgen = (np.sum(Qg) + Qi[slack] + Qd[slack]) * 1j

        if 'DS' in algorithm: # algorithm models the distributed slack
            Pmismatch = P_losses_line + np.real(S_shunt)
            Pgen = np.sum(Pg + K*Pmismatch)
        else:
            Pgen = np.sum(Pg) + Pi[slack] + Pd[slack]

    S_gen = (Pgen + Qgen) * 100
    S_load = (Pload + Qload) * 100
    S_mismatch = (P_losses_line + Q_losses_line + S_shunt) * 100

    if 'DS' in algorithm:
        return (Power_branches, S_gen, S_load, S_mismatch, Pmismatch)
    else:
        return (Power_branches, S_gen, S_load, S_mismatch)


def print_voltage_profile(V_polar_final,N):
    print("\n\tVoltage profile:")
    print("   Bus    Magnitude (p.u.)    Phase Angle (degrees)")
    if N <= 31:
        print(*("{:>6d}\t     {:1.6f}\t\t{:11.6f}" \
            .format(i,mag,ang) for i,(mag,ang) in enumerate(V_polar_final)), sep='\n')
    else:
        print(*("{:>6d}\t     {:1.6f}\t\t{:11.6f}" \
            .format(i,mag,ang) for i,(mag,ang) in enumerate(V_polar_final[0:14])), sep='\n')
        print(* 3*("     .\t         .\t\t      .",), sep='\n')
        print(*("{:>6d}\t     {:1.6f}\t\t{:11.6f}" \
            .format(i,mag,ang) for i,(mag,ang) in enumerate(V_polar_final[N-14:N],N-14)), sep='\n')
    print()


def create_power_balance_string(
    scale, Mis, algorithm,
    list_coef_or_iterations, S_gen, S_load, S_mismatch,
    Ploss=None, Pmismatch=None
):
    coef_or_iterations = 'Coefficients' if algorithm[0:2] == 'HE' else 'Iterations'
    output = \
        'Scale: {:d}   Mismatch: {}'.format(scale, Mis) + \
        '   {:s} per PVLIM-PQ switches: {:s}' \
            .format(coef_or_iterations, str(list_coef_or_iterations)) + \
        "\n\n  *  Power Balance:  *" + \
        "\n\nTotal generated power (MVA):  ----------------> {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_gen),np.imag(S_gen)) + \
        "\nTotal demanded power (MVA):  -----------------> {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_load),np.imag(S_load)) + \
        "\nTotal power through branches and shunt" + \
        "\nelements (mismatch) (MVA):  ------------------> {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_mismatch),np.imag(S_mismatch)) + \
        "\n\nComparison: Generated power (MVA):  ----------> {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_gen),np.imag(S_gen)) + \
        "\n            Demanded plus mismatch power (MVA): {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_load+S_mismatch),np.imag(S_load+S_mismatch))
    if Ploss is not None:
        output = output + \
        "\n\nComparison: Active power losses 'Ploss' variable (MW):  ---------------------> {:< 22.15f}" \
            .format(np.real(Ploss*100)) + \
        "\n            Active power through branches and shunt elements 'Pmismatch' (MW): {:< 22.15f}" \
            .format(np.real(Pmismatch*100))

    return output


def write_results_on_files(
    case, scale, Mis, algorithm,
    V_polar_final, V_complex_profile, Power_branches,
    list_coef_or_iterations, S_gen, S_load, S_mismatch,
    Ploss=None, Pmismatch=None
):
    case = basename(case)
    files_name = \
        'Results' + ' ' + \
        algorithm + ' ' + \
        str(case) + ' ' + \
        str(scale) + ' ' + \
        str(Mis)

    # Write voltage profile and branch data to .xlsx file
    voltages_dataframe = pd.DataFrame()
    voltages_dataframe["Complex Voltages"] = V_complex_profile
    voltages_dataframe["Voltages Magnitude"] = V_polar_final[:,0]
    voltages_dataframe["Voltages Phase Angle"] = V_polar_final[:,1]
    power_flow_dataframe = pd.DataFrame()
    power_flow_dataframe["From Bus"] = Power_branches[:,0]
    power_flow_dataframe["To Bus"] = Power_branches[:,1]
    power_flow_dataframe['From-To P injection (MW)'] = Power_branches[:,2]
    power_flow_dataframe['From-To Q injection (MVAR)'] = Power_branches[:,3]
    power_flow_dataframe['To-From P injection (MW)'] = Power_branches[:,4]
    power_flow_dataframe['To-From Q injection (MVAR)'] = Power_branches[:,5]
    power_flow_dataframe['P flow through branch and elements (MW)'] = Power_branches[:,6]
    power_flow_dataframe['Q flow through branch and elements (MVAR)'] = Power_branches[:,7]
    xlsx_name = files_name + '.xlsx'
    xlsx_file = pd.ExcelWriter(xlsx_name)
    voltages_dataframe.to_excel(xlsx_file, sheet_name="Buses")
    power_flow_dataframe.to_excel(xlsx_file, sheet_name="Branches")
    xlsx_file.save()

    # Write power balance and other data to .txt file
    # Coefficients/Iterations per PVLIM-PQ switches are written
    txt_content = create_power_balance_string(
        scale, Mis, algorithm,
        list_coef_or_iterations, S_gen, S_load, S_mismatch,
        Ploss, Pmismatch
    )
    txt_name = files_name + ".txt"
    txt_file = open(txt_name,"w")
    txt_file.write(txt_content)
    txt_file.close()

    print("\nResults have been written on the files:\n\t%s \n\t%s"%(xlsx_name,txt_name))


# Main loop
def helm_modified_ds_m1_pv1(
        grid_data_file_path,
        Print_Details=False, Mismatch=1e-4, Scale=1,
        MaxCoefficients=100, Enforce_Qlimits=True, DSB_model=True,
        Results_FileName='', Save_results=False,
):
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

    Qg, V_complex_profile, Vre, Vimag, list_gen_remove, list_coef  = initialize_data_arrays(N)
    
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
        K = compute_k_factor(Pg, Pd, slack, N, list_gen)
        # Set the slack's participation factor to 1 and the rest to 0. Classic slack bus model.
        if not(DSB_model):
            K_slack_1(K,slack)
        # Create modified Y matrix and list that contains the respective column to its voltage on PV and PVLIM buses 
        solve, Y_Vsp_PV = Modif_Ytrans(Ytrans, N, Buses_type, K, slack, branches_buses, list_gen)
        # Arrays and lists creation
        unknowns, coefficients, Soluc_no_eval, Soluc_eval = Unknowns_soluc(Buses_type, N, N_coef)
        # Loop of coefficients computing until the mismatch is reached
        Flag_recalculate, Flag_divergence, series_large = computing_voltages_mismatch( 
            Y_Vsp_PV, Soluc_no_eval, Soluc_eval, coefficients, N, Mis, 
            Pg, Pd, Qg, Qd, V_complex_profile, solve,
            list_coef, detailed_run_print, N_coef,
            Q_limits, Buses_type, Qgmax, Qgmin, Yre, Yimag, Vre, Vimag,
            branches_buses, list_gen_remove, list_gen, V,
            K, slack, Yshunt, phase_barras, phase_dict, # Function Gen_bus y otras
        )
        if Flag_recalculate:
            break
    if not Flag_divergence:
        if detailed_run_print or Save_results:
            Ploss = Pade(coefficients[2*N], series_large)
            Power_branches, S_gen, S_load, S_mismatch, Pmismatch = power_balance(
                V_complex_profile, Ybr_list,
                N_branches, N, Shunt, slack, Pd, Qd, Pg, Qg,
                Q_limits, list_gen,
                Vre, Vimag, Yre, Yimag , branches_buses, algorithm,
                K=K,
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
