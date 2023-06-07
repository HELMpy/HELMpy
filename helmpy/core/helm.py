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

from helmpy.core.classes import RunVariables, CaseData
from helmpy.core.analytic_continuation import Pade

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)

from numba import jit
from typing import Tuple

@jit
def modif_Ytrans(DSB_model_method, pv_bus_model, case, run):
    """Create modified Y matrix and list that contains the respective column 
    to its voltage on PV and PVLIM buses 
    """
    # Assign local variables for faster access
    N = case.N
    slack = case.slack
    Ytrans = case.Ytrans
    branches_buses = case.branches_buses
    Buses_type = run.Buses_type
    K = run.K

    # Declare new variables
    Ytrans_mod = np.zeros((run.length,run.length), dtype=float)
    Y_Vsp_PV =[]

    for i in range(N):
        if Buses_type[i] == 'Slack':
            Ytrans_mod[2*i][2*i] = 1
            Ytrans_mod[2*i + 1][2*i + 1] = 1
        elif Buses_type[i] == 'PQ' or pv_bus_model == 1:
            for j in branches_buses[i]:
                Ytrans_mod[2*i][2*j] = Ytrans[i][j].real
                Ytrans_mod[2*i][2*j + 1] = Ytrans[i][j].imag*-1
                Ytrans_mod[2*i + 1][2*j] = Ytrans[i][j].imag
                Ytrans_mod[2*i + 1][2*j + 1] = Ytrans[i][j].real
        else: # pv_bus_model == 2:
            Ytrans_mod[2*i + 1][2*i] = 1
            for j in branches_buses[i]:
                Ytrans_mod[2*i][2*j] = Ytrans[i][j].real
                Ytrans_mod[2*i][2*j + 1] = Ytrans[i][j].imag*-1

    if DSB_model_method is not None:
        # Last row
        for j in branches_buses[slack]:
            Ytrans_mod[2*N][2*j] = Ytrans[slack][j].real
            Ytrans_mod[2*N][2*j + 1] = Ytrans[slack][j].imag*-1
        # Last column
        for i in run.list_gen:
            Ytrans_mod[i*2][2*N] = -K[i]
        Ytrans_mod[2*N][2*N] = -K[slack]

    if pv_bus_model == 1:
        if DSB_model_method is None:
            for i in run.list_gen:
                array = np.zeros( 2*len(branches_buses[i]), dtype=np.float64)
                pos = 0
                for k in branches_buses[i]:
                    array[pos] = Ytrans_mod[2*k][2*i]
                    array[pos+1] = Ytrans_mod[2*k+1][2*i]
                    Ytrans_mod[2*k][2*i] = 0
                    Ytrans_mod[2*k+1][2*i] = 0
                    pos += 2
                Y_Vsp_PV.append([i, array.copy()])
                Ytrans_mod[2*i + 1][2*i] = 1
        else:
            for i in run.list_gen:
                if slack in branches_buses[i]:
                    array = np.zeros( 2*len(branches_buses[i])+1, dtype=np.float64)
                else:
                    array = np.zeros( 2*len(branches_buses[i]), dtype=np.float64)
                pos = 0
                for k in branches_buses[i]:
                    array[pos] = Ytrans_mod[2*k][2*i]
                    array[pos+1] = Ytrans_mod[2*k+1][2*i]
                    Ytrans_mod[2*k][2*i] = 0
                    Ytrans_mod[2*k+1][2*i] = 0
                    pos += 2
                if slack in branches_buses[i]:
                    array[pos] = Ytrans_mod[2*N][2*i]
                    Ytrans_mod[2*N][2*i] = 0
                Y_Vsp_PV.append([i, array.copy()])
                Ytrans_mod[2*i + 1][2*i] = 1
        run.Y_Vsp_PV = Y_Vsp_PV
    
    # Return a function for solving a sparse linear system, with Ytrans_mod pre-factorized.
    solve = factorized(csc_matrix(Ytrans_mod))
    run.solve = solve

@jit
def Unknowns_soluc(DSB_model_method, pv_bus_model, N, run):
    """Arrays and lists creation"""
    # Assign local variables for faster access
    coefficients = run.coefficients
    Soluc_no_eval = run.Soluc_no_eval
    Buses_type = run.Buses_type

    # Assign 0 to the first coefficients and evaluated solutions.
    coefficients[:,0].fill(0)
    run.Soluc_eval[:,0].fill(0)
    # Clear list of not evaluated solutions (function per bus)
    Soluc_no_eval.clear()

    for i in range(N):
        if Buses_type[i] == 'PV' or Buses_type[i] == 'PVLIM':
            if pv_bus_model == 1:
                Soluc_no_eval.append([i,evaluate_bus_eq_dsb_generator_pv1])
            else: # pv_bus_model == 2:
                coefficients[2*i][0] = 1
                Soluc_no_eval.append([i,evaluate_bus_eq_dsb_generator_pv2])
        else:
            coefficients[2*i][0] = 1
            if Buses_type[i] == 'PQ':
                Soluc_no_eval.append([i, evaluate_bus_eq_dsb_load])
            else: # Buses_type[i] == 'slack'
                Soluc_no_eval.append([i, evaluate_bus_eq_dsb_slack])
    if DSB_model_method == 1:
        Soluc_no_eval.append([N,evaluate_bus_eq_dsb_method1])
    elif DSB_model_method == 2:
        Soluc_no_eval.append([N,evaluate_bus_eq_dsb_method2])

@jit
def Calculo_Vre_PV(n, case, run):
    """Real voltage of PV and PVLIM buses computing.
    
    coefficient n
    """
    # Assign local variables for faster access
    V = case.V
    V_complex = run.V_complex
    Vre_PV = run.Vre_PV

    for i in run.list_gen:
        if n > 1:
            aux = 0
            for k in range(1,n):
                aux += V_complex[i][k] * np.conj(V_complex[i][n-k])
            Vre_PV[i][n] = -aux/2
        elif n == 1:
            Vre_PV[i][n] = (V[i]**2 - 1)/2

#---------------------------------------------------------------------------------------
# Functions lo evaluate the rigth hand side of the matrix equation
@jit
def evaluate_bus_eq_dsb_method1(_, n, Si, Pi, case, run):
    """Function to evaluate the PV bus equation for the slack bus by method 1
    
    coefficient n
    """
    # Assign local variables for faster access
    N = case.N
    phase_dict = case.phase_dict
    W = run.W
    V_complex = run.V_complex
    coefficients = run.coefficients
    i = case.slack

    aux_Ploss = 0
    for k in range(1,n):
        aux_Ploss += coefficients[N*2][k]*np.conj(W[i][n-k])
    
    PP = 0
    if case.phase_barras[i]:
        for k in range(len(phase_dict[i][0])):
            PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1]

    result = Pi[i]*np.conj(W[i][n-1]) - case.Yshunt[i]*V_complex[i][n-1] - PP + run.K[i]*aux_Ploss

    run.Soluc_eval[2*N][n] = np.real(result)


def evaluate_bus_eq_dsb_method2(_, n, Si, Pi, case, run):
    """Function to evaluate the PV bus equation for the slack bus by method 2
    
    coefficient n"""
    # Assign local variables for faster access
    Ytrans = case.Ytrans
    branches_buses = case.branches_buses
    phase_dict = case.phase_dict
    V_complex = run.V_complex
    i = case.slack
    if run.pv_bus_model == 2:
        slack_CC = run.barras_CC[i]
    else:
        slack_CC = run.slack_CC

    if n > 2:
        CC = 0
        PPP = 0
        for x in range(1, n-1):
            PPP += np.conj(V_complex[i][n-x]) * slack_CC[x]
        PP = 0
        for k in branches_buses[i]:
            PP += Ytrans[i][k] * V_complex[k][n-1]
        slack_CC[n-1] = PP
        PPP += np.conj(V_complex[i][1]) * PP
        CC -= PPP.real
        # Valor Shunt
        if case.conduc_buses[i]:
            VV = 0
            for k in range(1,n-1):
                VV += V_complex[i][k] * np.conj(V_complex[i][n-k])
            CC -= np.real(case.Yshunt[i]) * ( VV + 2*V_complex[i][n-1].real )
        # Valores phase
        if case.phase_barras[i]:
            PPP = 0
            for x in range(n):
                PP = 0
                for k in range(len(phase_dict[i][0])):
                    PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1-x]
                PPP += np.conj(V_complex[i][x]) * PP
            CC -= PPP.real
    elif n == 1:
        CC = Pi[i] - np.real(case.Yshunt[i])
        # Valores phase
        if case.phase_barras[i]:
            for valor in phase_dict[i][1]:
                CC -= valor.real
    elif n == 2:
        CC = 0
        PP = 0
        for k in branches_buses[i]:
            PP += Ytrans[i][k] * V_complex[k][1]
        slack_CC[1] = PP
        CC -= ( np.conj(V_complex[i][1]) * PP ).real
        # Valor Shunt
        if case.conduc_buses[i]:
            CC -= np.real(case.Yshunt[i])*2*V_complex[i][1].real
        # Valores phase
        if case.phase_barras[i]:
            PPP = 0
            for x in range(n):
                PP = 0
                for k in range(len(phase_dict[i][0])):
                    PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1-x]
                PPP += np.conj(V_complex[i][x]) * PP
            CC -= PPP.real

    run.Soluc_eval[2*case.N][n] = CC

@jit
def evaluate_bus_eq_dsb_generator_pv1(i, n, Si, Pi, case, run):
    """Function to evaluate the PV buses equation by py model 1
    
    bus i, coefficient n
    """
    # Assign local variables for faster access
    N = case.N
    phase_dict = case.phase_dict
    W = run.W
    V_complex = run.V_complex
    coefficients = run.coefficients

    aux = 0
    for k in range(1,n):
        aux += coefficients[i*2][k]*np.conj(W[i][n-k])
    
    aux_Ploss = 0
    if run.DSB_model_method is not None:
        for k in range(1,n):
            aux_Ploss += coefficients[N*2][k]*np.conj(W[i][n-k])
    
    PP = 0
    if case.phase_barras[i]:
        for k in range(len(phase_dict[i][0])):
            PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1]
    
    result = Pi[i]*np.conj(W[i][n-1]) - case.Yshunt[i]*V_complex[i][n-1]  - PP - aux*1j + run.K[i]*aux_Ploss
    
    run.Soluc_eval[2*i][n] = np.real(result)
    run.Soluc_eval[2*i + 1][n] = np.imag(result)

@jit
def evaluate_bus_eq_dsb_generator_pv2(i, n, Si, Pi, case, run):
    """Function to evaluate the PV buses equation by py model 2
    
    bus i, coefficient n
    """
    # Assign local variables for faster access
    Ytrans = case.Ytrans
    branches_buses = case.branches_buses
    phase_dict = case.phase_dict
    V_complex = run.V_complex
    barras_CC = run.barras_CC

    if n > 2:
        CC = 0
        PPP = 0
        for x in range(1,n-1):
            PPP += np.conj(V_complex[i][n-x]) * barras_CC[i][x]
        PP = 0
        for k in branches_buses[i]:
            PP += Ytrans[i][k] * V_complex[k][n-1]
        barras_CC[i][n-1] = PP
        PPP += np.conj(V_complex[i][1]) * PP
        CC -= PPP.real
        # Valor Shunt
        if case.conduc_buses[i]:
            CC -= np.real(case.Yshunt[i]) * ( run.VVanterior[i] + 2*V_complex[i][n-1].real )
        # Valores phase
        if case.phase_barras[i]:
            PPP = 0
            for x in range(n):
                PP = 0
                for k in range(len(phase_dict[i][0])):
                    PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1-x]
                PPP += np.conj(V_complex[i][x]) * PP
            CC -= PPP.real
    elif n == 2:
        CC = 0
        PP = 0
        for k in branches_buses[i]:
            PP += Ytrans[i][k] * V_complex[k][1]
        barras_CC[i][1] = PP
        CC -= ( np.conj(V_complex[i][1]) * PP ).real
        # Valor Shunt
        if case.conduc_buses[i]:
            CC -= np.real(case.Yshunt[i])*2*V_complex[i][1].real
        # Valores phase
        if case.phase_barras[i]:
            PPP = 0
            for x in range(n):
                PP = 0
                for k in range(len(phase_dict[i][0])):
                    PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1-x]
                PPP += np.conj(V_complex[i][x]) * PP
            CC -= PPP.real
    elif n == 1:
        CC = Pi[i] - np.real(case.Yshunt[i])
        # Valores phase
        if case.phase_barras[i]:
            for valor in phase_dict[i][1]:
                CC -= valor.real

    if n == 1:
        VV = case.V[i]**2 - 1
    else:
        VV = 0
        for k in range(1,n):
            VV += V_complex[i][k] * np.conj(V_complex[i][n-k])
        run.VVanterior[i] = VV
        VV = -VV

    run.Soluc_eval[2*i][n] = CC
    run.Soluc_eval[2*i + 1][n] = VV/2

@jit
def evaluate_bus_eq_dsb_load(i, n, Si, Pi, case, run):
    """Function to evaluate the PQ buses equation
    
    bus i, coefficient n
    """
    # Assign local variables for faster access
    phase_dict = case.phase_dict
    W = run.W
    V_complex = run.V_complex

    PP = 0
    if case.phase_barras[i]:
        for k in range(len(phase_dict[i][0])):
            PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1]
    
    result = np.conj(Si[i])*np.conj(W[i][n-1]) - case.Yshunt[i]*V_complex[i][n-1] - PP
    
    run.Soluc_eval[2*i][n] = np.real(result)
    run.Soluc_eval[2*i + 1][n] = np.imag(result)

@jit
def evaluate_bus_eq_dsb_slack(i, n, Si, Pi, case, run):
    """Function to evaluate the slack bus equation
    
    bus i, coefficient n
    """
    if n == 1:
        run.Soluc_eval[2*i][n] = case.V[i] - 1
        run.Soluc_eval[2*i + 1][n] = 0
    else:
        run.Soluc_eval[2*i][n] = 0
        run.Soluc_eval[2*i + 1][n] = 0

#---------------------------------------------------------------------------------------
@jit
def compute_complex_voltages(n, pv_bus_model, case, run):
    """Complex voltages computing
    
    coefficient n"""
    # Assign local variables for faster access
    Vre_PV = run.Vre_PV
    V_complex = run.V_complex
    coefficients = run.coefficients
    Buses_type = run.Buses_type

    if pv_bus_model == 2:
        for i in range(case.N):
            V_complex[i][n] = coefficients[i*2][n] + 1j*coefficients[i*2 + 1][n]
    else: # pv_bus_model == 1:
        for i in range(case.N):
            if Buses_type[i] == 'PV' or Buses_type[i] == 'PVLIM':
                V_complex[i][n] = Vre_PV[i][n] + 1j*coefficients[i*2 + 1][n]
            else:
                V_complex[i][n] = coefficients[i*2][n] + 1j*coefficients[i*2 + 1][n]

@jit
def calculate_inverse_voltages_w_array(n, case, run):
    """W computing - Inverse voltages "W" array"""
    # Assign local variables for faster access
    W = run.W
    V_complex = run.V_complex

    for i in range(case.N):
        aux = 0
        for k in range(n):
            aux += (W[i][k] * V_complex[i][n-k])
        W[i][n] = -aux

@jit
def P_iny(i, case, run):
    """Computing P injection at bus i. Must be used after Voltages_profile()"""
    # Assign local variables for faster access
    Yre = case.Yre
    Yimag = case.Yimag
    branches_buses = case.branches_buses
    Vre = run.Vre
    Vimag = run.Vimag

    Piny = 0
    for k in branches_buses[i]:
        Piny += Vre[i] * (Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) \
                + Vimag[i] * (Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    return Piny

@jit
def Q_iny(i, case, run):
    """Computing Q injection at bus i. Must be used after Voltages_profile()"""
    # Assign local variables for faster access
    Yre = case.Yre
    Yimag = case.Yimag
    branches_buses = case.branches_buses
    Vre = run.Vre
    Vimag = run.Vimag

    Qiny = 0
    for k in branches_buses[i]:
        Qiny += Vimag[i] * (Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) \
                - Vre[i] * (Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    return Qiny

@jit
def check_PVLIM_violation(detailed_run_print, case, run):
    """Verification of Qgen limits for PVLIM buses"""
    # Assign local variables for faster access
    Qd = case.Qd
    Qgmax = case.Qgmax
    Qgmin = case.Qgmin
    Qg = run.Qg
    list_gen = run.list_gen
    list_gen_remove = run.list_gen_remove
    Buses_type = run.Buses_type

    flag_violacion = False
    for i in list_gen:
        Qg_incog = Q_iny(i, case, run) + Qd[i]
        Qg[i] = Qg_incog
        if Qg_incog > Qgmax[i] or Qg_incog < Qgmin[i]:
            flag_violacion = True
            Buses_type[i] = 'PQ'
            list_gen_remove.append(i)
            Qg[i] = Qgmax[i] if Qg_incog > Qgmax[i] else Qgmin[i]
            if detailed_run_print:
                print('Bus %d exceeded its Qgen limit with %f. The exceeded limit %f will be assigned to the bus'%(i+1,Qg_incog,Qg[i]))
    return flag_violacion

@jit
def compute_k_factor(case, run):
    """Computing of the K factor for each PV bus and the slack bus.
    
    Only the PV buses are considered to calculate Pgen_total. The PV buses 
    that were converted to PQ buses are NOT considered.
    """
    # Assign local variables for faster access
    K = run.K
    Pg = run.Pg

    K.fill(0)
    Pgen_total = 0
    Distrib = []
    # Active power that the slack must generate to compensate the system
    Pg[case.slack] = run.Pg_imbalance
    for i in run.list_gen:
        if Pg[i] > 0:
            Pgen_total += Pg[i]
            Distrib.append(i)
    if Pg[case.slack] > 0:
        Pgen_total += Pg[case.slack]
        Distrib.append(case.slack)
    for i in Distrib:
        K[i] = Pg[i]/Pgen_total

@jit
def K_slack_1(case, run):
    """Set the slack's participation factor to 1 and the rest to 0. 
    
    Classic slack bus model.
    """
    run.K.fill(0)
    run.K[case.slack] = 1


def computing_voltages_mismatch(
    detailed_run_print, mismatch, max_coef, enforce_Q_limits,
    pv_bus_model, DSB_model_method, case, run
):
    """Loop of coefficients computing until the mismatch is reached"""
    # Assign local variables for faster access
    slack = case.slack
    branches_buses = case.branches_buses
    Soluc_no_eval = run.Soluc_no_eval
    N = case.N
    Y_Vsp_PV = run.Y_Vsp_PV
    list_coef = run.list_coef
    solve = run.solve
    V_complex_profile = run.V_complex_profile
    Soluc_eval = run.Soluc_eval
    coefficients = run.coefficients
    resta_columnas_PV = run.resta_columnas_PV
    Vre_PV = run.Vre_PV
    V_complex = run.V_complex
    
    # Variables initialization
    coef_actual = 0
    series_large = 1
    run.W[:,0] = 1 # Assign 1 to the inverse voltages of coefficients 0

    # Compute Vre_PV and V_complex for coefficient 0
    if pv_bus_model == 1:
        Vre_PV[:,0] = 1
    compute_complex_voltages(0, pv_bus_model, case, run)

    # Compute active and complex power injection
    Pi = run.Pg - case.Pd
    Si = Pi + run.Qg*1j - case.Qd*1j

    # Flags
    first_check = True
    flag_recalculate = False
    flag_divergence = False

    while True:
        coef_actual += 1
        if coef_actual == 40:
            # Expand the coeffcients arrays to the maximum. They were originally set to 40
            run.expand_coef_arrays()
        if detailed_run_print:
            print("Computing coefficient: %d"%coef_actual)

        # Compute Vre_PV for current coefficient. Only for pv_bus_model 1 
        if pv_bus_model == 1:
            Calculo_Vre_PV(coef_actual, case, run)

        # Compute the right hand side of the matrix equation
        for i in range(len(Soluc_no_eval)):
            Soluc_no_eval[i][1](Soluc_no_eval[i][0], coef_actual, Si, Pi, case, run)

        # Determine right_hand_side of matrix equation
        if pv_bus_model == 1:
            # Columns to subtract
            resta_columnas_PV.fill(0)
            for Vre_vec in Y_Vsp_PV:
                array = Vre_PV[Vre_vec[0]][coef_actual] * Vre_vec[1]
                pos = 0
                for k in branches_buses[Vre_vec[0]]:
                    resta_columnas_PV[2*k] += array[pos]
                    resta_columnas_PV[2*k+1] += array[pos+1]
                    pos += 2
                if DSB_model_method is not None:
                    if slack in branches_buses[Vre_vec[0]]:
                        resta_columnas_PV[2*N] += array[pos]
            right_hand_side = Soluc_eval[:,coef_actual] - resta_columnas_PV
        else: # pv_bus_model == 2:
            right_hand_side = Soluc_eval[:,coef_actual]

        # New column of coefficients
        coefficients[:,coef_actual] = solve(right_hand_side)

        # Compute V_complex and inverse voltages for current coefficient 
        compute_complex_voltages(coef_actual, pv_bus_model, case, run)
        calculate_inverse_voltages_w_array(coef_actual, case, run)
        
        # Mismatch check
        flag_mismatch = False
        series_large += 1
        if (series_large - 1) % 2 == 0 and series_large > 3:
            if first_check:
                first_check = False
                for i in range(N):
                    magn1,rad1 = cm.polar(Pade(V_complex[i],series_large-2))
                    V_complex_profile[i] = Pade(V_complex[i],series_large)
                    magn2, rad2 = cm.polar(V_complex_profile[i])
                    if((abs(magn1-magn2)>mismatch) or (abs(rad1-rad2)>mismatch)):
                        flag_mismatch = True
                        pade_til = i+1
                        break
            else:
                continue_check = True
                for i in range(pade_til):
                    magn1,rad1 = cm.polar(V_complex_profile[i])
                    V_complex_profile[i] = Pade(V_complex[i],series_large)
                    magn2, rad2 = cm.polar(V_complex_profile[i])
                    if((abs(magn1-magn2)>mismatch) or (abs(rad1-rad2)>mismatch)):
                        flag_mismatch = True
                        pade_til = i+1
                        continue_check = False
                        break
                if continue_check:
                    for i in range(pade_til,N):
                        magn1,rad1 = cm.polar(Pade(V_complex[i],series_large-2))
                        V_complex_profile[i] = Pade(V_complex[i],series_large)
                        magn2, rad2 = cm.polar(V_complex_profile[i])
                        if((abs(magn1-magn2)>mismatch) or (abs(rad1-rad2)>mismatch)):
                            flag_mismatch = True
                            pade_til = i+1
                            break
            if not flag_mismatch:
                # Qgen check or ignore limits
                if enforce_Q_limits:
                    if check_PVLIM_violation(detailed_run_print, case, run):
                        if detailed_run_print:
                            print("At coefficient %d the system is to be resolved due to PVLIM to PQ switches\n"%series_large)
                        list_coef.append(series_large)
                        flag_recalculate = True
                        break
                print('\nConvergence has been reached. %d coefficients were calculated'%series_large)
                list_coef.append(series_large)
                break
        if series_large > max_coef-1:
            print('\nMaximum number of coefficients has been reached. The problem has no physical solution')
            flag_divergence = True
            break
    
    return flag_recalculate, flag_divergence, series_large

@jit
def convert_complex_to_polar_voltages(complex_voltage, N):
    """Separate each voltage value in magnitude and phase angle (degrees)"""
    polar_voltage = np.empty((N,2), dtype=np.float64)
    polar_voltage[:,0] = np.absolute(complex_voltage)
    polar_voltage[:,1] = np.angle(complex_voltage, deg=True)
    return polar_voltage


def power_balance(enforce_Q_limits, algorithm, case, run):
    """Computation of power flow through branches and power balance"""
    # Save for later: Pi=None, Qi=None, K=None 

    # Assign local variables for faster access
    Ybr_list = case.Ybr_list
    Shunt = case.Shunt
    slack = case.slack
    Pd = case.Pd
    Qd = case.Qd
    K = run.K
    V_complex_profile = run.V_complex_profile
    Pg = run.Pg
    Qg = run.Qg
    list_gen = run.list_gen

    # Define array to power flow through branches data
    Power_branches = np.zeros((case.N_branches,8), dtype=np.float64)

    for branch in range(case.N_branches):

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
    for i in range(case.N):
        if Shunt[i] != 0:
            S_shunt += V_complex_profile[i] * np.conj(V_complex_profile[i]*Shunt[i])

    Qload = np.sum(Qd) * 1j
    Pload = np.sum(Pd)

    if 'HELM' in algorithm:

        if not enforce_Q_limits:
            for i in list_gen:
                Qg[i] = Q_iny(i, case, run) + Qd[i]
        Qgen = (np.sum(Qg) + Q_iny(slack, case, run) + Qd[slack]) * 1j

        if 'DS' in algorithm: # algorithm models the distributed slack
            Pmismatch = P_losses_line + np.real(S_shunt)
            Pgen = np.sum(Pg + K*Pmismatch)
        else:
            Pgen = np.sum(Pg) + P_iny(slack, case, run) + Pd[slack]

    # elif 'NR' in algorithm:

    #     if not enforce_Q_limits:
    #         for i in list_gen:
    #             Qg[i] = Qi[i] + Qd[i]
    #     Qgen = (np.sum(Qg) + Qi[slack] + Qd[slack]) * 1j

    #     if 'DS' in algorithm: # algorithm models the distributed slack
    #         Pmismatch = P_losses_line + np.real(S_shunt)
    #         Pgen = np.sum(Pg + K*Pmismatch)
    #     else:
    #         Pgen = np.sum(Pg) + Pi[slack] + Pd[slack]

    S_gen = (Pgen + Qgen) * 100
    S_load = (Pload + Qload) * 100
    S_mismatch = (P_losses_line + Q_losses_line + S_shunt) * 100

    if 'DS' in algorithm:
        return (Power_branches, S_gen, S_load, S_mismatch, Pmismatch)
    else:
        return (Power_branches, S_gen, S_load, S_mismatch, None)

@jit
def print_voltage_profile(V_polar_final, N):
    """Print voltage profile."""
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

@jit
def create_power_balance_string(
    mismatch, scale, algorithm,
    list_coef_or_iterations, S_gen, S_load, S_mismatch,
    Ploss=None, Pmismatch=None
):
    coef_or_iterations = 'Coefficients' if algorithm[0:2] == 'HE' else 'Iterations'
    output = \
        'Scale: {}   Mismatch: {}'.format(scale, mismatch) + \
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

@jit
def write_results_on_files(
    mismatch, scale, algorithm,
    V_polar_final, Power_branches,
    results_file_name, run,
    power_balance_string,
):
    files_name = \
        'Results' + ' ' + \
        algorithm + ' ' + \
        str(results_file_name) + ' ' + \
        str(scale) + ' ' + \
        str(mismatch)

    # Write voltage profile and branch data to .xlsx file
    voltages_dataframe = pd.DataFrame()
    voltages_dataframe["Complex Voltages"] = run.V_complex_profile
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
    xlsx_file = pd.ExcelWriter(xlsx_name) # pylint: disable=abstract-class-instantiated
    voltages_dataframe.to_excel(xlsx_file, sheet_name="Buses")
    power_flow_dataframe.to_excel(xlsx_file, sheet_name="Branches")
    xlsx_file.save()

    # Write power balance and other data to .txt file
    # Coefficients/Iterations per PVLIM-PQ switches are written
    txt_name = files_name + ".txt"
    txt_file = open(txt_name,"w")
    txt_file.write(power_balance_string)
    txt_file.close()

    print("\nResults have been written on the files:\n\t%s \n\t%s"%(xlsx_name,txt_name))

@jit
def validate_arguments(
        case,
        detailed_run_print, mismatch, scale,
        max_coefficients, enforce_Q_limits,
        results_file_name, save_results,
        pv_bus_model, DSB_model, DSB_model_method, 
):
    if (type(detailed_run_print) is not bool or \
        type(mismatch) is not float or \
        not(
            type(scale) is float or
            type(scale) is int
        ) or \
        type(max_coefficients) is not int or \
        type(enforce_Q_limits) is not bool or \
        not(
            results_file_name is None or
            type(results_file_name) is str
        ) or \
        type(save_results) is not bool or \
        type(pv_bus_model) is not int or \
        type(DSB_model) is not bool or \
        not(
            DSB_model_method is None or
            type(DSB_model_method) is int
        )
    ):
        print("Erroneous argument type.")
        return False, None
    if max_coefficients < 5:
        print("'max_coefficients' must be equal or greater than five (5).")
        return False, None
    if pv_bus_model not in (1, 2):
        print("'pv_bus_model' must be the integer 1 or 2.",)
        return False, None
    if DSB_model_method is not None and DSB_model_method not in (1, 2):
        print("'DSB_model_method' must be the integer 1 or 2.",)
        return False, None

    return True


# Main loop
def helm(case, detailed_run_print=False, mismatch=1e-4, scale=1, max_coefficients=100, enforce_Q_limits=True,
         results_file_name=None, save_results=False, pv_bus_model=2, DSB_model=False, DSB_model_method=None,
         ) -> Tuple[RunVariables, int, bool]:

    # Arguments validation
    if not validate_arguments(case, detailed_run_print, mismatch, scale, max_coefficients, enforce_Q_limits,
                              results_file_name, save_results, pv_bus_model, DSB_model, DSB_model_method):
        raise ValueError('Arguments were wrong.')

    if DSB_model and DSB_model_method is None:
        DSB_model_method = 2

    #  Construct algorithm string
    pv_bus_model_str = 'PV1' if pv_bus_model == 1 else 'PV2'
    if DSB_model_method is not None:
        DSB_model_method_str = 'M1' if DSB_model_method == 1 else 'M2'
        algorithm = 'HELM DS ' + DSB_model_method_str + ' ' + pv_bus_model_str
    else:
        algorithm = 'HELM ' + pv_bus_model_str

    if results_file_name is None:
        results_file_name = case.name

    max_coef = max_coefficients

    # set case at the scale
    if scale != 1:
        case.set_scale(scale)

    # Declare run_variables_class objects.
    # Variables/arrays initialization are inside it
    run = RunVariables(case, pv_bus_model, DSB_model, DSB_model_method, max_coef)

    while True:
        # Re-construct list_gen. List of generators (PV buses)
        run.list_gen = np.setdiff1d(run.list_gen, run.list_gen_remove, assume_unique=True)

        # Define K factors
        if DSB_model:
            # Computing the K factor for each PV bus and the slack bus.
            compute_k_factor(case, run)
        elif DSB_model_method is not None:
            # Set the slack's participation factor to 1 and the rest to 0. Classic slack bus model.
            K_slack_1(case, run)

        # Create modified Y matrix and list that contains the respective column to its voltage on PV and PVLIM buses 
        modif_Ytrans(DSB_model_method, pv_bus_model, case, run)

        # Arrays and lists creation
        Unknowns_soluc(DSB_model_method, pv_bus_model, case.N, run)

        # Loop of coefficients computing until the mismatch is reached
        flag_recalculate, flag_divergence, series_large = computing_voltages_mismatch(detailed_run_print, mismatch,
                                                                                      max_coef, enforce_Q_limits,
                                                                                      pv_bus_model, DSB_model_method,
                                                                                      case, run)

        if not flag_recalculate:
            break
    # reset scale case
    if scale != 1:
        case.reset_scale()

    if not flag_divergence:
        if detailed_run_print or save_results:
            Ploss = None

            if DSB_model_method is not None:
                Ploss = Pade(run.coefficients[2*case.N], series_large)

            Power_branches, S_gen, S_load, S_mismatch, Pmismatch = power_balance(enforce_Q_limits, algorithm, case, run)

            if detailed_run_print or save_results:
                V_polar_final = convert_complex_to_polar_voltages(run.V_complex_profile, case.N)

                power_balance_string = create_power_balance_string(mismatch, scale, algorithm, run.list_coef, S_gen,
                                                                   S_load, S_mismatch, Ploss, Pmismatch)
                if detailed_run_print:
                    print_voltage_profile(V_polar_final, case.N)
                    print(power_balance_string)
                if save_results:
                    write_results_on_files(mismatch, scale, algorithm, V_polar_final, Power_branches, results_file_name,
                                           run, power_balance_string)

    return run, series_large, flag_divergence
