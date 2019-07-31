"""
HELMpy, open source package of power flow solvers developed on Python 3 
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix
import cmath as cm
import pandas as pd
import warnings

from helmpy.core.nr import get_case_name_from_path_without_extension
from helmpy.core.write_results_to_csv import write_results_to_csv
from helmpy.util.root_path import ROOT_PATH

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)


#Global Variables that will be the basic parameters of the main function
detailed_run_print = False   #Print details on the run
Mis = 1e-4      #Mismatch
case = ''       #results file name
scale = 1       #load scale
N_coef = 100    #Max number of coefficients
Q_limits = True     #Checks Q generation limits

#Global variables declaration
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
barras_CC = dict()
VVanterior = np.zeros(N, dtype=float)
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
def Dimension():
    global V, Pg, Qg, Pd, Qd, Buses_type, Qgmax, Qgmin, N, Y, Yshunt, Shunt
    global Ytrans, Pg_sch, Number_bus, VVanterior
    global N_branches, branches_buses, phase_barras
    global Pi, conduc_buses, Yre, Yimag, V_complex_profile, list_gen, V_polar_final
    global phase_dict, barras_CC, first_check, list_gen_remove, list_coef, Power_branches, Ybr_list
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
    barras_CC = dict()
    VVanterior = np.zeros(N, dtype=float)
    V_complex_profile = np.zeros(N, dtype=complex)
    list_gen = np.zeros(N_generators-1, dtype=int)
    V_polar_final = np.zeros((N,2), dtype=float)
    first_check = True
    list_gen_remove = []
    list_coef = []
    Ybr_list = list()
    Power_branches = np.zeros((N_branches,8), dtype=float)

# Branches data processing to construct Ytrans, Yshunt, branches_buses and others
def branches_processor(i, FromBus, ToBus, R, X, BTotal, Tap, Shift_degree):
    global Ytrans, Yshunt, Number_bus, branches_buses, Ybr_list
    global phase_barras, phase_dict
    FB = Number_bus[FromBus] 
    TB = Number_bus[ToBus]
    Ybr_list.append([FB, TB, np.zeros((2,2),dtype=complex)])
    Z = R + 1j*X
    if(Tap==0 or Tap==1):
        if(Z!=0):
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
        if(Z!=0):
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


    if( TB not in branches_buses[FB] ):
        branches_buses[FB].append(TB)
    if( FB not in branches_buses[TB] ):
        branches_buses[TB].append(FB)

# Processing of .xls file data
def Buses_xls():
    global Buses_type, V, Qgmax, Qgmin, Pd, Qd, Pg, Shunt, buses, branches, N_branches
    global slack_bus, N, N_generators, generators, Number_bus, Yshunt, Ytrans
    global Y, conduc_buses, Pg_sch, Yre, Yimag, slack, list_gen, num_gen
    global scale

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
    Pg_sch = np.copy(Pg)

    for i in range(N_branches):
        branches_processor(i, branches[0][i], branches[1][i], branches[2][i], branches[3][i], branches[4][i], branches[8][i], branches[9][i])

    for i in range(N):
        branches_buses[i].sort()    # Variable that saves the branches
        barras_CC[i] = np.zeros(N_coef, dtype=complex)    # Variable that saves the calculated values of PV buses

    Y = Ytrans.copy()
    for i in range(N):
        if( Yshunt[i].real != 0 ):
            conduc_buses[i] = True
        Y[i,i] += Yshunt[i]
        if(phase_barras[i]):
            for k in range(len(phase_dict[i][0])):
                Y[i,phase_dict[i][0][k]] += phase_dict[i][1][k]
    Yre = np.real(Y)
    Yimag = np.imag(Y)

# Modified Y matrix
Ytrans_mod = np.zeros((2*N+1,2*N+1),dtype=float)
# Create modified Y matrix
def Modif_Ytrans():
    global Ytrans, Ytrans_mod, N, Buses_type, K, slack, solve, branches_buses, list_gen
    Ytrans_mod = np.zeros((2*N+1,2*N+1),dtype=float)

    for i in range(N):
        if(Buses_type[i]=='Slack'):
            Ytrans_mod[2*i][2*i]=1
            Ytrans_mod[2*i + 1][2*i + 1]=1
        elif(Buses_type[i]=='PQ'):
            for j in branches_buses[i]:
                Ytrans_mod[2*i][2*j]=Ytrans[i][j].real
                Ytrans_mod[2*i][2*j + 1]=Ytrans[i][j].imag*-1
                Ytrans_mod[2*i + 1][2*j]=Ytrans[i][j].imag
                Ytrans_mod[2*i + 1][2*j + 1]=Ytrans[i][j].real
        elif(Buses_type[i]=='PVLIM' or Buses_type[i]=='PV'):
            Ytrans_mod[2*i + 1][2*i]=1
            for j in branches_buses[i]:
                Ytrans_mod[2*i][2*j]=Ytrans[i][j].real
                Ytrans_mod[2*i][2*j + 1]=Ytrans[i][j].imag*-1

    # Last row
    for k in branches_buses[slack]:
        Ytrans_mod[2*N][2*k] = Ytrans[slack][k].real
        Ytrans_mod[2*N][2*k+1] = Ytrans[slack][k].imag*-1

    # last Column
    for i in list_gen:
        Ytrans_mod[i*2][2*N]=-K[i]
    Ytrans_mod[2*N][2*N]=-K[slack]

    # Return a function for solving a sparse linear system, with Ytrans_mod pre-factorized.
    solve = factorized(csc_matrix(Ytrans_mod))

# List of unknowns
unknowns = []
# Coefficients array
coefficients = np.zeros((2*N+1,N_coef), dtype=float)
# Evaluated solutions columns array
Soluc_eval = np.zeros((2*N+1,N_coef), dtype=float)
# Functions list to evaluate
Soluc_no_eval = []
# Arrays and lists creation
def Unknowns_soluc():
    global Buses_type, unknowns, N, N_coef, coefficients, Soluc_no_eval, Soluc_eval
    unknowns = []
    coefficients = np.zeros((2*N+1,N_coef), dtype=float)
    Soluc_eval = np.zeros((2*N+1,N_coef), dtype=float)
    Soluc_no_eval = []
    for i in range(N):
        unknowns.append([i,'Vre'])
        coefficients[2*i][0] = 1
        unknowns.append([i,'Vim'])
        coefficients[2*i + 1][0] = 0
        if(Buses_type[i]=='PV' or Buses_type[i]=='PVLIM'):
            Soluc_no_eval.append([i,Gen_bus2])
        elif(Buses_type[i]=='PQ'):
            Soluc_no_eval.append([i,Load_bus])
        else:
            Soluc_no_eval.append([i,Slack_bus])
    Soluc_no_eval.append([N,P_Loss2])
    unknowns.append([N,'PLoss'])
    coefficients[2*N][0] = 0

# Actualized complex voltages array of each bus (Vre+Vim sum)
V_complex = np.zeros((N,N_coef), dtype=complex)
# Complex voltages computing
def Voltages_complex(n):  # coefficient n
    global V_complex, coefficients, Buses_type, N
    for i in range(N):
        V_complex[i][n] = coefficients[i*2][n] + 1j*coefficients[i*2 + 1][n]

# Inverse voltages "W" array
W = np.ones((N,N_coef), dtype=complex)
# W computing
def Calculate_W(n):
    global W, V_complex, N
    if(n>0):
        for i in range(N):
            aux = 0
            for k in range(n):
                aux += (W[i][k] * V_complex[i][n-k])
            W[i][n] = -aux

# Function to evaluate the PV bus equation for the slack bus 
def P_Loss2(nothing,n):  # coefficient n
    global Soluc_eval, V_complex, coefficients, Ytrans, Pi, branches_buses, barras_CC
    global phase_barras, phase_dict, conduc_buses, slack, Yshunt
    i = slack

    if(n>2):
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
        #Valor Shunt
        if(conduc_buses[i]):
            CC -= np.real(Yshunt[i]) * ( VVanterior[i] + 2*V_complex[i][n-1].real )
        #Valores phase
        if(phase_barras[i]):  
            PPP = 0
            for x in range(n):
                PP = 0
                for k in range(len(phase_dict[i][0])):
                    PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1-x]
                PPP += np.conj(V_complex[i][x]) * PP
            CC -= PPP.real
    elif(n==1):
        CC = Pi[i] - np.real(Yshunt[i])
        #Valores phase
        if(phase_barras[i]):
            for valor in phase_dict[i][1]:
                CC -= valor.real
    elif(n==2):
        CC = 0
        PP = 0
        for k in branches_buses[i]:
            PP += Ytrans[i][k] * V_complex[k][1]
        barras_CC[i][1] = PP
        CC -= ( np.conj(V_complex[i][1]) * PP ).real
        #Valor Shunt
        if(conduc_buses[i]):
            CC -= np.real(Yshunt[i])*2*V_complex[i][1].real
        #Valores phase
        if(phase_barras[i]):
            PPP = 0
            for x in range(n):
                PP = 0
                for k in range(len(phase_dict[i][0])):
                    PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1-x]
                PPP += np.conj(V_complex[i][x]) * PP
            CC -= PPP.real

    Soluc_eval[2*N][n] = CC

# Function to evaluate the slack bus equation 
def Slack_bus(i,n):  # bus i, coefficient n
    global V, Soluc_eval
    if(n==1):
        Soluc_eval[2*i][n] = V[i] - 1

# Function to evaluate the PQ buses equation 
def Load_bus(i,n):  # bus i, coefficient n
    global Pg, Pd, Qg, Qd, Si, Soluc_eval, W, Yshunt, V_complex, phase_barras
    global phase_dict
    if(phase_barras[i]):
        PP = 0
        for k in range(len(phase_dict[i][0])):
            PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1]
        result = np.conj(Si[i])*np.conj(W[i][n-1]) - Yshunt[i]*V_complex[i][n-1] - PP
    else:
        result = np.conj(Si[i])*np.conj(W[i][n-1]) - Yshunt[i]*V_complex[i][n-1]
    Soluc_eval[2*i][n] = np.real(result)
    Soluc_eval[2*i + 1][n] = np.imag(result)

# Function to evaluate the PV buses equation 
def Gen_bus2(i,n):  # bus i, coefficient n
    global Pg, Pd, Soluc_eval, V_complex, coefficients, V, Ytrans, Pi, branches_buses
    global barras_CC, VVanterior, phase_barras, phase_dict, conduc_buses

    if(n>2):
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
        #Valor Shunt
        if(conduc_buses[i]):
            CC -= np.real(Yshunt[i]) * ( VVanterior[i] + 2*V_complex[i][n-1].real )
        #Valores phase
        if(phase_barras[i]):  
            PPP = 0
            for x in range(n):
                PP = 0
                for k in range(len(phase_dict[i][0])):
                    PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1-x]
                PPP += np.conj(V_complex[i][x]) * PP
            CC -= PPP.real
    elif(n==1):
        CC = Pi[i] - np.real(Yshunt[i])
        #Valores phase
        if(phase_barras[i]):
            for valor in phase_dict[i][1]:
                CC -= valor.real
    elif(n==2):
        CC = 0
        PP = 0
        for k in branches_buses[i]:
            PP += Ytrans[i][k] * V_complex[k][1]
        barras_CC[i][1] = PP
        CC -= ( np.conj(V_complex[i][1]) * PP ).real
        #Valor Shunt
        if(conduc_buses[i]):
            CC -= np.real(Yshunt[i])*2*V_complex[i][1].real
        #Valores phase
        if(phase_barras[i]):
            PPP = 0
            for x in range(n):
                PP = 0
                for k in range(len(phase_dict[i][0])):
                    PP += phase_dict[i][1][k] * V_complex[phase_dict[i][0][k]][n-1-x]
                PPP += np.conj(V_complex[i][x]) * PP
            CC -= PPP.real

    if(n==1):
        VV = V[i]**2 - 1
    else:
        VV = 0
        for k in range(1,n):
            VV += V_complex[i][k] * np.conj(V_complex[i][n-k])
        VVanterior[i] = VV
        VV = -VV

    Soluc_eval[2*i][n] = CC
    Soluc_eval[2*i + 1][n] = VV/2

# Epsilon Algorithm for the analytic continuation of the power series through Padé approximants
def Epsilon(serie_completa,largo_actual):
    serie = serie_completa[:largo_actual]
    cont = np.zeros((len(serie),len(serie)+1), dtype=complex)
    cont[0][1] = serie[0]
    for i in range(1,len(serie)):
        cont[i][1] = cont[i-1][1] + serie[i] 
    for col in range(2,len(serie)+1):
        for row in range(0,len(serie)+1-col):
            cont[row][col] = cont[row+1][col-2] + 1/(cont[row+1][col-1]-cont[row][col-1])
    return cont[0][len(serie)]

# Matrix method for the analytic continuation of the power series through Padé approximants
def Pade(serie_completa,largo_actual):
    serie = serie_completa[:largo_actual]
    L = int((len(serie)-1)/2)
    mat_c = np.zeros((L,L), dtype=complex)
    for i in range(1,L+1):
        for j in range(i,L+i):
            mat_c[i-1][j-i] = serie[j]
    vec_b = -np.linalg.solve(mat_c,serie[L+1:len(serie)])
    b = np.ones(L+1, dtype=complex)
    for i in range(1,L+1):
        b[i] = vec_b[L-i]
    a = np.zeros(L+1, dtype=complex)
    a[0] = serie[0]
    for i in range(1,L+1):
        aux = 0
        for k in range(i+1):
            aux += serie[k]*b[i-k]
        a[i] = aux
    return np.sum(a)/np.sum(b)

# Separation of the voltage profile in its real and imaginary parts
def Voltages_profile():
    global V_complex_profile, Vre, Vimag
    Vre = np.real(V_complex_profile)
    Vimag = np.imag(V_complex_profile)

# Verification of Qgen limits for PVLIM buses
def Check_PVLIM():
    global Buses_type, Qg, Qd, Qgmax, Qgmin, N, num_gen, list_gen_remove, list_gen, detailed_run_print
    flag_violacion = 0
    Voltages_profile()

    for i in list_gen:
        Qg_incog = Q_iny(i) + Qd[i]
        Qg[i] = Qg_incog
        if((Qg_incog>Qgmax[i]) or (Qg_incog<Qgmin[i])):
            flag_violacion = 1
            Buses_type[i] = 'PQ'
            num_gen -= 1
            list_gen_remove.append(i)
            if(Qg_incog>Qgmax[i]):
                Qg[i] = Qgmax[i]
            else:
                Qg[i] = Qgmin[i]
            if(detailed_run_print):
                print('Bus %d exceeded its Qgen limit with %f. The exceeded limit %f will be assigned to the bus'%(i+1,Qg_incog,Qg[i]))
    return flag_violacion

# Computing Q injection at bus i. Must be used after Voltages_profile()
def Q_iny(i):
    global Vre, Vimag, Yre, Yimag, N
    Qiny = 0
    for k in branches_buses[i]:
        Qiny += Vimag[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) - Vre[i]*(Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    return Qiny

# Re-construct list_gen. List of generators (PV buses)
def Make_list_gen():
    global num_gen, list_gen, list_gen_remove
    list_gen_aux = np.zeros(num_gen, dtype=int)
    if(len(list_gen_remove)!=0):
        pos = 0
        for i in list_gen:
            if(i not in list_gen_remove):
                list_gen_aux[pos] = i
                pos += 1
        list_gen = list_gen_aux.copy()

# Vector of the participation factors: K's
K = np.zeros(N, dtype=float)
Distrib = []
# Computing of the K factor for each PV bus and the slack bus.
# Only the PV buses are considered to calculate Pgen_total. The PV buses that were converted to PQ buses are NOT considered.
def Compute_K():
    global K, Pg, Buses_type, slack, N, Pg_sch, Pd, Distrib, list_gen
    Pgen_total = 0
    K = np.zeros(N, dtype=float)
    Distrib = []
    Pg = np.copy(Pg_sch)
    # Active power that the slack must generate to compensate the system
    Pg[slack] = np.sum(Pd) - np.sum(Pg)
    for i in list_gen:
        if(Pg[i]>0):
            Pgen_total += Pg[i]
            Distrib.append(i)
    if(Pg[slack]>0):
        Pgen_total += Pg[slack]
        Distrib.append(slack)
    for i in Distrib:
        K[i] = Pg[i]/Pgen_total

# Set the slack's participation factor to 1 and the rest to 0. Classic slack bus model.
def K_slack_1():
    global K, slack
    K = np.zeros(N, dtype=float)
    K[slack] = 1

# Series coefficients counter (to mismatch) 
series_large = 0
# Loop of coefficients computing until the mismatch is reached
def Computing_Voltages_Mismatch():
    global series_large, Soluc_no_eval, Vre_PV, Soluc_eval, coefficients, N, Mis
    global Ytrans_mod, solve,V_complex, W, Pi, Si, Pg, Pd, Qg, Qd
    global V_complex_profile, first_check, pade_til, list_coef, detailed_run_print, Flag_divergence, Q_limits
    V_complex = np.zeros((N,N_coef), dtype=complex)
    W = np.ones((N,N_coef), dtype=complex)
    Flag_recalculate = 1
    Flag_divergence = False
    Voltages_complex(0)
    Pi = Pg - Pd
    Si = Pi + Qg*1j - Qd*1j
    coef_actual = 0
    series_large = 1
    while(1):
        coef_actual += 1
        if(detailed_run_print):
            print("Computing coefficient: %d"%coef_actual)

        for i in range(len(Soluc_no_eval)):
            Soluc_no_eval[i][1](Soluc_no_eval[i][0],coef_actual)

        # New column of coefficients
        coefficients[:,coef_actual] =solve(Soluc_eval[:,coef_actual])

        Voltages_complex(coef_actual)
        Calculate_W(coef_actual)

        # Mismatch check
        Flag_Mismatch = 0
        series_large += 1
        if( ((series_large-1)%2==0) and (series_large > 3) ):
            if(first_check):
                first_check = False
                for i in range(N):
                    magn1,rad1 = cm.polar(Pade(V_complex[i],series_large-2))
                    V_complex_profile[i] = Pade(V_complex[i],series_large)
                    magn2,rad2 = cm.polar(V_complex_profile[i])
                    if((abs(magn1-magn2)>Mis) or (abs(rad1-rad2)>Mis)):
                        Flag_Mismatch = 1
                        pade_til = i+1
                        break
            else:
                seguir = True
                for i in range(pade_til):
                    magn1,rad1 = cm.polar(V_complex_profile[i])
                    V_complex_profile[i] = Pade(V_complex[i],series_large)
                    magn2,rad2 = cm.polar(V_complex_profile[i])
                    if((abs(magn1-magn2)>Mis) or (abs(rad1-rad2)>Mis)):
                        Flag_Mismatch = 1
                        pade_til = i+1
                        seguir = False
                        break
                if(seguir):
                    for i in range(pade_til,N):
                        magn1,rad1 = cm.polar(Pade(V_complex[i],series_large-2))
                        V_complex_profile[i] = Pade(V_complex[i],series_large)
                        magn2,rad2 = cm.polar(V_complex_profile[i])
                        if((abs(magn1-magn2)>Mis) or (abs(rad1-rad2)>Mis)):
                            Flag_Mismatch = 1
                            pade_til = i+1
                            break
            if(Flag_Mismatch==0):
                # Qgen check or ignore limits
                if(Q_limits):
                    if(Check_PVLIM()):
                        if(detailed_run_print):
                            print("At coefficient %d the system is to be resolved due to PVLIM to PQ switches\n"%series_large)
                        list_coef.append(series_large)
                        Flag_recalculate = 0
                        return Flag_recalculate
                print('\nConvergence has been reached. %d coefficients were calculated'%series_large)
                list_coef.append(series_large)
                break
        if(series_large > N_coef-1):
            print('\nThe problem has no physical solution')
            Flag_divergence = 1
            break
    return Flag_recalculate

# Computation of power flow trough branches and power balance
def Power_balance():
    global V_complex_profile, Ybr_list, Power_branches, N_branches, Power_print, N, Shunt, slack, Pd, Qd, Pg, Qg, K, Pmismatch, S_gen, S_load, S_mismatch, Ploss, detailed_run_print, Q_limits, list_gen


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

    Power_print = pd.DataFrame()
    Power_print["From Bus"] = Power_branches[:,0]
    Power_print["To Bus"] = Power_branches[:,1]
    Power_print['From-To P injection (MW)'] = Power_branches[:,2]
    Power_print['From-To Q injection (MVAR)'] = Power_branches[:,3]
    Power_print['To-From P injection (MW)'] = Power_branches[:,4]
    Power_print['To-From Q injection (MVAR)'] = Power_branches[:,5]
    Power_print['P flow through branch and elements (MW)'] = Power_branches[:,6]
    Power_print['Q flow through branch and elements (MVAR)'] = Power_branches[:,7]
    P_losses_line = np.sum(Power_branches[:,6])/100
    Q_losses_line = np.sum(Power_branches[:,7]) * 1j /100
    

    # Computation of power through shunt capacitors, reactors or conductantes, Power balanca
    S_shunt = 0
    for i in range(N):
        if(Shunt[i]!=0):
            S_shunt += V_complex_profile[i] * np.conj(V_complex_profile[i]*Shunt[i])

    Pmismatch = P_losses_line + np.real(S_shunt)

    Pload   = np.sum(Pd)
    Pgen = 0
    for i in range(N):
        Pgen  += Pg[i] + K[i]*Pmismatch

    Qload   = np.sum(Qd) * 1j
    if not(Q_limits):
        Voltages_profile()
        for i in list_gen:
            Qg[i] = Q_iny(i) + Qd[i]
    Qgen    = (np.sum(Qg) + Q_iny(slack) + Qd[slack]) * 1j

    S_gen = (Pgen + Qgen) * 100
    S_load = (Pload + Qload) * 100
    S_mismatch = (P_losses_line + Q_losses_line + S_shunt) * 100

    if(detailed_run_print):
        print("\n\n\tPower balance:\nTotal generated power (MVA):\t\t\t\t\t\t\t"+str(np.real(S_gen))+" + "+str(np.imag(S_gen))+"j\nTotal demanded power (MVA):\t\t\t\t\t\t\t"+str(np.real(S_load))+" + "+str(np.imag(S_load))+"j\nTotal power through branches and shunt elements (mismatch) (MVA):\t\t"+str(np.real(S_mismatch))+" + "+str(np.imag(S_mismatch))+"j")
        print("\nComparison between generated power and demanded plus mismatch power (MVA):\t"+str(np.real(S_gen))+" + "+str(np.imag(S_gen))+"j  =  "+str(np.real(S_load+S_mismatch))+" + "+str(np.imag(S_load+S_mismatch))+"j")
        print("\nComparison between active power losses 'Ploss' and active power\nthrough branches and shunt elements 'Pmismatch' (MW):\t\t\t\t"+str(np.real(Ploss*100))+" = "+str(Pmismatch*100))

# Separate each voltage value in magnitude and phase angle (degrees). Calculate Ploss
V_polar_final = np.zeros((N,2), dtype=float)
def Final_Results():
    global V_complex_profile, V_polar_final, N, Ploss
    for i in range(N):
        magnitude, radians = cm.polar(V_complex_profile[i])
        V_polar_final[i,0],V_polar_final[i,1] = magnitude, np.rad2deg(radians)
    Ploss = Pade(coefficients[2*N],series_large)

def Print_Voltage_Profile():
    global V_polar_final, N, detailed_run_print
    if(detailed_run_print):
        print("\n\tVoltage profile:")
        print("   Bus    Magnitude (p.u.)    Phase Angle (degrees)")
        if(N<=31):
            for i in range(N):
                print("%6s"%i,"\t     %1.6f"%V_polar_final[i,0],"\t\t{:11.6f}".format(V_polar_final[i,1]))
        else:
            for i in range(14):
                print("%6s"%i,"\t     %1.6f"%V_polar_final[i,0],"\t\t{:11.6f}".format(V_polar_final[i,1]))
            print("     .\t         .\t\t      .")
            print("     .\t         .\t\t      .")
            print("     .\t         .\t\t      .")
            for i in range(N-14,N):
                print("%6s"%i,"\t     %1.6f"%V_polar_final[i,0],"\t\t{:11.6f}".format(V_polar_final[i,1]))


def write_results_on_files():
    global V_polar_final, T, Mis, scale, list_coef, case, V_complex_profile, Power_print, Pmismatch, S_gen, S_load, S_mismatch, Ploss
    # Write voltage profile to csv file
    data = pd.DataFrame()
    data["Complex Voltages"] = V_complex_profile
    data["Voltages Magnitude"] = V_polar_final[:,0]
    data["Voltages Phase Angle"] = V_polar_final[:,1]
    case = get_case_name_from_path_without_extension(case)

    write_results_to_csv(
        Mis, Power_print, case, data, scale,
        algorithm='HELM DS M2 PV2',
    )

    # Coefficients per PVLIM-PQ switches are written on a .txt file
    txt_name = "HELM DS M2 PV2 "+str(case)+' '+str(scale)+' '+str(Mis)+".txt"
    result = open(ROOT_PATH / 'data' / 'txt' / txt_name,"w")
    result.write('Scale:'+str(scale)+'\tMismatch:'+str(Mis)+'\n'+'Coefficients per PVLIM-PQ switches: '+str(list_coef))
    result.write("\n\nPower balance:\n\nTotal generated power (MVA):\t\t\t\t\t\t\t"+str(np.real(S_gen))+" + "+str(np.imag(S_gen))+"j\nTotal demanded power (MVA):\t\t\t\t\t\t\t"+str(np.real(S_load))+" + "+str(np.imag(S_load))+"j\nTotal power through branches and shunt elements (mismatch) (MVA):\t\t"+str(np.real(S_mismatch))+" + "+str(np.imag(S_mismatch))+"j")
    result.write("\n\nComparison between generated power and demanded plus mismatch power (MVA):\t"+str(np.real(S_gen))+" + "+str(np.imag(S_gen))+"j  =  "+str(np.real(S_load+S_mismatch))+" + "+str(np.imag(S_load+S_mismatch))+"j")
    result.write("\n\nComparison between active power losses 'Ploss' and active power\nthrough branches and shunt elements 'Pmismatch' (MW):\t\t\t\t"+str(np.real(Ploss*100))+" = "+str(Pmismatch*100))
    result.close()
    print("\nResults have been written on the files:\n\t%s"%(txt_name))


# Main loop
def helm_ds_m2_pv2(
        *,
        Print_Details=False, Mismatch=1e-4, Results_FileName='', Scale=1, MaxCoefficients=100, Enforce_Qlimits=True, DSB_model=True,
        generators_file_path, buses_file_path, branches_file_path,
):
    global V_complex_profile, N, buses, branches, N_branches, N_coef, N_generators, generators, T, Flag_divergence
    global detailed_run_print, Mis, case, scale, N_coef, Q_limits
    if (type(Print_Details)is not bool) or(type(Mismatch)is not float) or(type(Results_FileName)is not str) or not( (type(Scale)is float) or(type(Scale)is int) ) or(type(MaxCoefficients) is not int) or(type(DSB_model) is not bool) or(type(Enforce_Qlimits) is not bool):
        print("Erroneous argument type.")
        return

    detailed_run_print = Print_Details
    Mis = Mismatch
    case = generators_file_path[0:-len('.csv')]
    scale = Scale
    N_coef = MaxCoefficients
    Q_limits = Enforce_Qlimits

    generators = pd.read_csv(generators_file_path, header=None)
    buses = pd.read_csv(buses_file_path, header=None)
    branches = pd.read_csv(branches_file_path, header=None)

    N = len(buses.index)
    N_generators = len(generators.index)
    N_branches = len(branches.index)

    Dimension()
    Buses_xls()
    while(1):
        # Re-construct list_gen. List of generators (PV buses)
        Make_list_gen()
        # Computing of the K factor for each PV bus and the slack bus.
        Compute_K()
        # Set the slack's participation factor to 1 and the rest to 0. Classic slack bus model.
        if not(DSB_model):
            K_slack_1()
        # Create modified Y matrix and list that contains the respective column to its voltage on PV and PVLIM buses 
        Modif_Ytrans()
        # Arrays and lists creation
        Unknowns_soluc()
        # Loop of coefficients computing until the mismatch is reached
        if(Computing_Voltages_Mismatch()):
            break
    if not(Flag_divergence):
        Final_Results() # Separate each voltage value in magnitude and phase angle (degrees). Calculate Ploss
        Print_Voltage_Profile()
        Power_balance()
        write_results_on_files()
        return V_complex_profile
