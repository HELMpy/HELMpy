
"""
HELMpy, open source package of power flow solvers developed on Python 3 
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""


# Import libraries and functions
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import cmath as cm
from time import time
import warnings

from helmpy.root_path import ROOT_PATH

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)


#Global Variables that will be the basic parameters of the main function
detailed_run_print = False   #Print details on the run
Mis = 1e-4      #Mismatch
case = ''       #results file name
scale = 1       #load scale
iterations_limit = 15  #Max number of iterations per convergence check
Q_limits = True     #Checks Q generation limits

#Global variables declaration
N = 0; ref = 0; Y = 0;  Buses = [];
Yshunt = []; Ytrans = []; V = []; tita = []; tita_degree = []; iterations = 0; 
Pg = np.zeros(N); Pesp = np.copy(Pg); Qg = []; Pd = []; Qgmax = [];
Qgmin = []; Qd = []; Shunt = []; dimension = 0;
Pg_total = 0;  Qg_total = 0;  Pd_total = 0;
deltas_P_Q = 0; Jaco = 0; known = []; unknown = []; PVLIM_flag = 0; PVLIM_buses = False
deltas_tita_V = []; Plin_totales = 0; Qlin_totales = 0; Slin_total = 0
Qg_total_2 = 0; Ilineas = []; divergence = False
buses = 0
branches = 0
N_branches = 0
N_generators = 0
generators = 0
Number_bus = dict()
Ytrans_unsy = np.zeros((N,N), dtype=complex)
jaco_funct_store = []
Gen_contribute = []
Vre  = np.ones(N, dtype=float)
Vimag = np.zeros(N, dtype=float)
V_complex_profile = np.ones(N, dtype=complex)
Yre = np.zeros((N,N), dtype=float)
Yimag = np.zeros((N,N), dtype=float)
Pi = np.zeros(N, dtype=float)
Qi = np.zeros(N, dtype=float)
branches_buses = []
T_bucle_out = 0
known_dict = dict() 
unknown_dict = dict()
list_iterations = []
Power_branches = np.zeros((N_branches,8), dtype=float)
Ybr_list = list()
Power_print = pd.DataFrame()
Pmismatch = 0
S_gen = 0
S_load = 0
S_mismatch = 0
list_gen = np.zeros(1, dtype=int)


def variables_initialization():
    global N, ref, Buses, V, tita, Pg, Pesp, Qg, Pd, Qd, Qgmax, Qgmin, Shunt, dimension, Y
    global deltas_P_Q, Jaco, known, unknown, PVLIM_flag, PVLIM_buses, iterations
    global Yshunt, Ytrans, Number_bus
    global Ytrans_unsy, jaco_funct_store, Gen_contribute, Vre, Vimag, V_complex_profile
    global Yre, Yimag, Pi, Qi, branches_buses, list_iterations, Power_branches, Ybr_list, list_gen

    dimension = 0
    deltas_P_Q = 0
    Jaco = 0
    known = []
    unknown = []
    PVLIM_flag = 0
    iterations = 0
    list_iterations = []
    ref = 0
    PVLIM_buses = False
    Buses = [0 for i in range(N)]
    V = np.ones(N)
    tita = np.zeros(N)
    Pg = np.zeros(N)
    Qg = np.zeros(N)
    Pd = np.zeros(N)
    Qd = np.zeros(N)
    Qgmax = np.zeros(N)
    Qgmin = np.zeros(N)
    Y = np.zeros((N,N), dtype=complex)
    Shunt = np.zeros(N, dtype=complex)
    Yshunt = np.zeros((N,N), dtype=complex)
    Ytrans = np.zeros((N,N), dtype=complex)
    Pesp = np.copy(Pg)
    Number_bus = dict()
    Ytrans_unsy = np.zeros((N,N), dtype=complex)
    jaco_funct_store = []
    Gen_contribute = []
    Vre  = np.ones(N, dtype=float)
    Vimag = np.zeros(N, dtype=float)
    V_complex_profile = np.ones(N, dtype=complex)
    Yre = np.zeros((N,N), dtype=float)
    Yimag = np.zeros((N,N), dtype=float)
    Pi = np.zeros(N, dtype=float)
    Qi = np.zeros(N, dtype=float)
    branches_buses = [[i] for i in range(N)]
    Ybr_list = list()
    Power_branches = np.zeros((N_branches,8), dtype=float)
    list_gen = np.zeros(N_generators-1, dtype=int)


# Branches data processing to construct Ytrans, Yshunt, branches_buses and others
def Branches_processor(i, FromBus, ToBus, R, X, BTotal, Tap, Shift_degree):
    global Ytrans, Yshunt, Number_bus, branches_buses, Ybr_list, Ytrans_unsy
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
                Ytrans_unsy[FB][TB] += Yseries_ft - Yseries_ft_shift
                Ytrans_unsy[TB][FB] += Yseries_ft - Yseries_tf_shift
            Ytrans[FB][TB] += -Yseries_ft
            Ytrans[FB][FB] +=  Yseries_ft
            Ytrans[TB][FB] += -Yseries_ft
            Ytrans[TB][TB] +=  Yseries_ft
        else:
            Ybr_list[i][2][0,1] = Ybr_list[i][2][1,0] = Yseries_ft = 0

        Bshunt_ft = 1j*BTotal/2
        Ybr_list[i][2][0,0] = Ybr_list[i][2][1,1] = Bshunt_ft + Yseries_ft
        Yshunt[FB][FB] +=  Bshunt_ft
        Yshunt[TB][TB] +=  Bshunt_ft
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
                Ytrans_unsy[FB][TB] += Yseries_ft - Yseries_ft_shift
                Ytrans_unsy[TB][FB] += Yseries_ft - Yseries_tf_shift
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
        Yshunt[FB][FB] +=  Bshunt_f - Yseries_ft
        Yshunt[TB][TB] +=  Bshunt_t - Yseries_ft


    if( TB not in branches_buses[FB] ):
        branches_buses[FB].append(TB)
    if( FB not in branches_buses[TB] ):
        branches_buses[TB].append(FB)


# Processing of .xlsx file data
def Buses_xls():
    global Buses, V, Qgmax, Qgmin, Pd, Qd, Pg, Shunt, buses, branches, N_branches
    global ref, N, N_generators, generators, Number_bus, Yshunt, Ytrans
    global Y, Ytrans_unsy, xls_actual, PVLIM_buses, Pesp, PVLIM_flag
    global Yre, Yimag, V_complex_profile, Vre, Vimag, scale, branches_buses, list_gen
    
    Pd = buses[2]/100*scale
    Qd = buses[3]/100*scale
    Shunt = buses[5]*1j/100 + buses[4]/100

    for i in range(N):
        Number_bus[buses[0][i]] = i
        if(buses[1][i]!=3):
            Buses[i] = 'PQ'
        else:
            ref = buses[0][i]
            aux_ref = i
        Yshunt[i][i] =  Shunt[i]

    pos = 0
    for i in range(N_generators):
        bus_i = Number_bus[generators[0][i]]
        if(bus_i!=aux_ref):
            list_gen[pos] = bus_i
            pos += 1
            Buses[bus_i] = 'PVLIM'
        PVLIM_buses = 1

        PVLIM_flag += 1
        V[bus_i] = generators[5][i]
        Pg[bus_i] = generators[1][i]/100*scale
        Qgmax[bus_i] = generators[3][i]/100
        Qgmin[bus_i] = generators[4][i]/100
        V_complex_profile[bus_i] = cm.rect(V[bus_i],tita[bus_i])
        Vre[bus_i]   = np.real(V_complex_profile[bus_i])
        Vimag[bus_i] = np.imag(V_complex_profile[bus_i])
       
    Buses[Number_bus[ref]] = 'Reference'
    Pg[Number_bus[ref]] = 0
    Pesp = np.copy(Pg)

    for i in range(N_branches):
        Branches_processor(i, branches[0][i], branches[1][i], branches[2][i], branches[3][i], branches[4][i], branches[8][i], branches[9][i])

    for i in range(N):
        branches_buses[i].sort()

    Y = Ytrans + Yshunt + Ytrans_unsy
    Yre = np.real(Y)
    Yimag = np.imag(Y)


# Structure and dimensions of the jacobian
def Jacobian():
    global Buses, known, unknown, dimension, deltas_P_Q, Jaco, branches_buses
    global known_dict, unknown_dict, Number_bus, ref
    dimension = 0
    known = []
    unknown = []
    known_dict = dict()
    unknown_dict = dict()
    
    pos_known = 0
    pos_unknown = 0
    slack = Number_bus[ref] 

    for i in range(N):
        if(Buses[i]!='Reference'):
            known.append(['dP',i])
            unknown.append(['dtita',i])

            known_dict[i] = [['dP', pos_known]]
            pos_known += 1
            unknown_dict[i] = [['dtita', pos_unknown]]
            pos_unknown += 1
            
            dimension += 1
    for i in range(N):
        if(Buses[i]=='PQ'):
            known.append(['dQ',i])
            unknown.append(['dV',i])

            known_dict[i].append(['dQ', pos_known])
            pos_known +=1
            unknown_dict[i].append(['dV',pos_unknown])
            pos_unknown += 1
            dimension += 1
    
    known_dict[slack] = [['nada', 0]]   #not used
    unknown_dict[slack] = [['nada', 0]]   #not used


    deltas_P_Q = np.zeros(dimension,dtype=float)
    Jaco = np.zeros((dimension,dimension),dtype=float)


# Store the corresponding function of each entry of the jacobian
def Jacobian_Functions():
    global jaco_funct_store, Jaco, branches_buses, known_dict, unknown_dict
    jaco_funct_store = []

    for i in range(N):
        bus_i = i

        for h in range(len(known_dict[bus_i])):
            delta_cono = known_dict[bus_i][h][0]
            row_jaco = known_dict[bus_i][h][1]

            for m in range(len(branches_buses[bus_i])):
                bus_j = branches_buses[bus_i][m]

                for k in range(len(unknown_dict[bus_j])):
                    delta_incog = unknown_dict[bus_j][k][0]
                    colum_jaco = unknown_dict[bus_j][k][1]

                    if((delta_cono=='dP') and (delta_incog=='dtita')):
                        if(bus_i==bus_j):
                           jaco_funct_store.append([Hii,bus_i,bus_j,row_jaco,colum_jaco])
                        else:
                            jaco_funct_store.append([Hik,bus_i,bus_j,row_jaco,colum_jaco])
                            
                    elif((delta_cono=='dP') and (delta_incog=='dV')):
                        if(bus_i==bus_j):
                            jaco_funct_store.append([Nii,bus_i,bus_j,row_jaco,colum_jaco])
                        else:
                            jaco_funct_store.append([Nik,bus_i,bus_j,row_jaco,colum_jaco])
                            
                    elif((delta_cono=='dQ') and (delta_incog=='dtita')):
                        if(bus_i==bus_j):
                            jaco_funct_store.append([Jii,bus_i,bus_j,row_jaco,colum_jaco])
                        else:
                            jaco_funct_store.append([Jik,bus_i,bus_j,row_jaco,colum_jaco])
                            
                    elif((delta_cono=='dQ') and (delta_incog=='dV')):
                        if(bus_i==bus_j):
                            jaco_funct_store.append([Lii,bus_i,bus_j,row_jaco,colum_jaco])
                        else:
                            jaco_funct_store.append([Lik,bus_i,bus_j,row_jaco,colum_jaco])


def Compute_Iterative_Jacobian_Entries():
    global Jaco, jaco_funct_store
    # Compute jacobian entries
    for i in range(len(jaco_funct_store)):
        Funct_actual = jaco_funct_store[i][0]
        bus_i = jaco_funct_store[i][1]
        bus_j = jaco_funct_store[i][2]
        position_i = jaco_funct_store[i][3]
        position_j = jaco_funct_store[i][4]
        Jaco[position_i][position_j] = Funct_actual(bus_i,bus_j)


def Convergence_Check():
    global known, deltas_P_Q, Mis, iterations, detailed_run_print
    global divergence, Pi, Qi, Pd, Qd, Pg, Qg, N, iterations_limit

    stop_iterations = False
    divergence = False

    for i in range(N):
        Piny(i)
        Qiny(i)

    # Computing delta P and Q
    for i in range(dimension):
        bus_i = known[i][1]
        if(known[i][0]=='dP'):
            deltas_P_Q[i] = Pg[bus_i] - Pd[bus_i] - Pi[bus_i]
        else:
            deltas_P_Q[i] = Qg[bus_i] - Qd[bus_i] - Qi[bus_i]


    reached_error = True
    error_max = max(abs(deltas_P_Q))
    if(error_max > Mis):
        if(detailed_run_print):
            print("Maximum error:", error_max)
        reached_error = False
    else:
        if(detailed_run_print):
            print("Program converged with a maximum error of:", error_max)

    if(reached_error):
        stop_iterations = True

    if( not(stop_iterations) ):
        iterations += 1
    if(iterations==iterations_limit+1):
        print("\nIteration number: %d. It is assumed that the program diverged."%(iterations-1))
        stop_iterations = True
        divergence = True
    if( detailed_run_print and not(stop_iterations) ):
        if(detailed_run_print):
            print("\nIteration number: %d"%(iterations))

    return stop_iterations


# Voltages, phase angles and Ploss results actualization on each iteration
def Actualizacion_Resultados():
    global unknown, deltas_tita_V, dimension, tita, V, V_complex_profile, Vre, Vimag, N
    
    result = []    
    for i in range(dimension):
        result.append([unknown[i][0],unknown[i][1],deltas_tita_V[i]])

    for j in range(dimension):
        if(result[j][0]=='dtita'):
            tita[result[j][1]] += result[j][2]
        else:
            V[result[j][1]] = V[result[j][1]]*(1 + result[j][2])

    for i in range(N):
        V_complex_profile[i] = cm.rect(V[i],tita[i])
        Vre[i]   = np.real(V_complex_profile[i])
        Vimag[i] = np.imag(V_complex_profile[i])


def Check_Generators_Limits():
    global PVLIM_buses, PVLIM_flag, Buses, Qg, Qgmax, Qgmin
    global N, detailed_run_print, Qi, iterations, list_iterations

    list_iterations.append(iterations)
    iterations = 0
    restart_NR = False
    if(PVLIM_buses):
        if(detailed_run_print):
            print('\nChecking PVLIM buses reactive power Qg limits')
        if(PVLIM_flag > 0):
            for i in range(N):
                if(Buses[i]=='PVLIM'):
                    Qg[i] = Qi[i] + Qd[i]
                    Qg_anterior = Qg[i]

                    if( Qg[i]>Qgmax[i] or Qg[i]<Qgmin[i]):
                        PVLIM_flag = PVLIM_flag - 1
                        restart_NR = True
                        Buses[i] = 'PQ'
                        if(Qg[i]>Qgmax[i]):
                            Qg[i] = Qgmax[i]
                        else:
                            Qg[i] = Qgmin[i]
                        if(detailed_run_print):
                            print('PVLIM bus %d exceeded its reactive power generation limit at %f MVAR. Exceeded limit: %f MVAR'%(i,Qg_anterior*100,Qg[i]*100))
            
    return restart_NR


# Functions to compute the jacobian entries
def Hii(i,nada):
    global Qi, V, Yimag
    Hii = -Qi[i] - (V[i]*V[i]*Yimag[i][i])
    return Hii


def Hik(i,k):
    global Vre, Yre, Yimag, Vimag 
    Hik = Vimag[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) - Vre[i]*(Yimag[i][k]*Vre[k] + Yre[i][k]*Vimag[k])
    return Hik


def Nii(i,nada):
    global Pi, V, Yre
    Nii = Pi[i] + V[i]*V[i]*Yre[i][i]
    return Nii


def Nik(i,k):
    global Vre, Yre, Yimag, Vimag
    Nik = Vre[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) + Vimag[i]*(Yimag[i][k]*Vre[k] + Yre[i][k]*Vimag[k])
    return Nik


def Jii(i,nada):
    global Pi, V, Yre
    Jii = Pi[i] - (V[i]*V[i]*Yre[i][i])
    return Jii


def Jik(i,k):
    global Vre, Yre, Yimag, Vimag
    Jik = -Vre[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) - Vimag[i]*(Yimag[i][k]*Vre[k] + Yre[i][k]*Vimag[k])
    return Jik


def Lii(i,nada):
    global Qi, V, Yimag
    Lii = Qi[i] - (V[i]*V[i]*Yimag[i][i])
    return Lii


def Lik(i,k):
    global Vre, Yre, Yimag, Vimag
    Lik = Vimag[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) - Vre[i]*(Yimag[i][k]*Vre[k] + Yre[i][k]*Vimag[k])
    return Lik


# Functions to compute power injection
def Piny(i):
    global Vre, Vimag, Yre, Yimag, Pi, N, branches_buses
    P_iny = 0
    for k in branches_buses[i]:
        P_iny += Vre[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) + Vimag[i]*(Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    Pi[i] = P_iny


def Qiny(i):
    global Vre, Vimag, Yre, Yimag, Qi, N, branches_buses
    Q_iny = 0
    for k in branches_buses[i]:
        Q_iny += Vimag[i]*(Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) - Vre[i]*(Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    Qi[i] = Q_iny


# Computation of power flow trough branches and power balance
def Power_balance():
    global V_complex_profile, Ybr_list, Power_branches, N_branches, Power_print, N, Shunt, Pd, Qd, Pg, Qg, K, Pmismatch, S_gen, S_load, S_mismatch, detailed_run_print, Pi, Qi, ref, Number_bus, Q_limits, list_gen

    slack = Number_bus[ref]
    
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

    Pload   = np.sum(Pd)
    Pgen = np.sum(Pg) + Pi[slack] + Pd[slack]

    Qload   = np.sum(Qd) * 1j
    if not(Q_limits):
        for i in list_gen:
            Qg[i] = Qi[i] + Qd[i]
    Qgen    = (np.sum(Qg) + Qi[slack] + Qd[slack]) * 1j
    

    S_gen = (Pgen + Qgen) * 100
    S_load = (Pload + Qload) * 100
    S_mismatch = (P_losses_line + Q_losses_line + S_shunt) * 100

    if(detailed_run_print):
        print("\n\n\tPower balance:\nTotal generated power (MVA):\t\t\t\t\t\t\t"+str(np.real(S_gen))+" + "+str(np.imag(S_gen))+"j\nTotal demanded power (MVA):\t\t\t\t\t\t\t"+str(np.real(S_load))+" + "+str(np.imag(S_load))+"j\nTotal power through branches and shunt elements (mismatch) (MVA):\t\t"+str(np.real(S_mismatch))+" + "+str(np.imag(S_mismatch))+"j")
        print("\nComparison between generated power and demanded plus mismatch power (MVA):\t"+str(np.real(S_gen))+" + "+str(np.imag(S_gen))+"j  =  "+str(np.real(S_load+S_mismatch))+" + "+str(np.imag(S_load+S_mismatch))+"j")


def Print_Voltage_Profile():
    global V, tita_degree, N, detailed_run_print
    if(detailed_run_print):
        print("\n\tVoltage profile:")
        print("   Bus    Magnitude (p.u.)    Phase Angle (degrees)")
        if(N<=31):
            for i in range(N):
                print("%6s"%i,"\t     %1.6f"%V[i],"\t\t{:11.6f}".format(tita_degree[i]))
        else:
            for i in range(14):
                print("%6s"%i,"\t     %1.6f"%V[i],"\t\t{:11.6f}".format(tita_degree[i]))
            print("     .\t         .\t\t      .")
            print("     .\t         .\t\t      .")
            print("     .\t         .\t\t      .")
            for i in range(N-14,N):
                print("%6s"%i,"\t     %1.6f"%V[i],"\t\t{:11.6f}".format(tita_degree[i]))


def write_results_on_files():
    global V_complex_profile, V, tita_degree, scale, T_bucle_out, list_iterations, Mis, case, Power_print, Pmismatch, S_gen, S_load, S_mismatch
    # Voltage profile is written on a .xlsx file
    data = pd.DataFrame()
    data["Complex Voltages"] = V_complex_profile
    data["Voltages Magnitude"] = V
    data["Voltages Phase Angle"] = tita_degree
    case = get_case_name_from_path_without_extension(case)
    xlsx_file_name = 'Results NR ' + \
                     str(case) + ' ' + \
                     str(scale) + ' ' + \
                     str(Mis) + '.xlsx'
    xlsx_file_path = ROOT_PATH / 'data' / 'results' / xlsx_file_name
    file = pd.ExcelWriter(xlsx_file_path)
    data.to_excel(file,sheet_name="Buses")
    # Branch info is written on .xlsx file
    Power_print.to_excel(file,sheet_name="Branches")
    file.save()
    # time and iterations are written on a .txt file
    txt_name = "NR "+str(case)+' '+str(scale)+' '+str(Mis)+".txt"
    result = open(ROOT_PATH / 'data' / 'txt' / txt_name,"w")
    result.write('Scale: '+str(scale)+'\tTime: '+str(T_bucle_out)+' sg'+'\tMismatch: '+str(Mis)+'\nIterations per PVLIM-PQ switches: '+str(list_iterations))
    result.write("\n\nPower balance:\n\nTotal generated power (MVA):\t\t\t\t\t\t\t"+str(np.real(S_gen))+" + "+str(np.imag(S_gen))+"j\nTotal demanded power (MVA):\t\t\t\t\t\t\t"+str(np.real(S_load))+" + "+str(np.imag(S_load))+"j\nTotal power through branches and shunt elements (mismatch) (MVA):\t\t"+str(np.real(S_mismatch))+" + "+str(np.imag(S_mismatch))+"j")
    result.write("\n\nComparison between generated power and demanded plus mismatch power (MVA):\t"+str(np.real(S_gen))+" + "+str(np.imag(S_gen))+"j  =  "+str(np.real(S_load+S_mismatch))+" + "+str(np.imag(S_load+S_mismatch))+"j")
    result.close()
    print("\nResults have been written on the files:\n\t%s \n\t%s"%(xlsx_file_path,txt_name))


def get_case_name_from_path_without_extension(case):
    """
    Remove absolute path prefix from case file

    :param case:
    :return: cleaned case name
    """

    prefix = ROOT_PATH / 'data' / 'case'
    return case[len(str(prefix)) + 1:]


# main function
def nr(GridName, Print_Details=False, Mismatch=1e-4, Results_FileName='', Scale=1, MaxIterations=15, Enforce_Qlimits=True):
    global Jaco, deltas_P_Q, deltas_tita_V, tita_degree, T_bucle_out
    global buses, branches, generators, N, N_generators, N_branches
    global detailed_run_print, Mis, case, scale, divergence, iterations_limit, Q_limits, list_iterations, iterations, V_complex_profile
    if (type(GridName)is not str) or(type(Print_Details)is not bool) or(type(Mismatch)is not float) or(type(Results_FileName)is not str) or not( (type(Scale)is float) or(type(Scale)is int) ) or(type(MaxIterations) is not int) or(type(Enforce_Qlimits) is not bool):
        print("Erroneous argument type.")
        return
    xls_actual = ROOT_PATH / 'data' / 'case' / GridName
    detailed_run_print = Print_Details
    Mis = Mismatch
    if(Results_FileName==''):
        case = GridName[0:-5]
    else:
        case = Results_FileName
    scale = Scale
    iterations_limit = MaxIterations
    Q_limits = Enforce_Qlimits
    
    buses = pd.read_excel(xls_actual, sheet_name='Buses', header=None)
    branches = pd.read_excel(xls_actual, sheet_name='Branches', header=None)
    generators = pd.read_excel(xls_actual, sheet_name='Generators', header=None)
    N = len(buses.index)
    N_generators = len(generators.index)
    N_branches = len(branches.index)

    variables_initialization()
    Buses_xls()
    # Loop that stops when the deltas P and Q be less than the specified mismatch, or the program diverges
    T_bucle_in = time()
    while(True):
        Jacobian()
        Jacobian_Functions()
        while(True):
            if( Convergence_Check() ): # Check convergence and iterations number
                break # Stop iterations

            Compute_Iterative_Jacobian_Entries()

            deltas_tita_V = spsolve(Jaco,deltas_P_Q)

            Actualizacion_Resultados()
        if(divergence):
            break
        if not(Q_limits):
            print("Convergence has been reached")
            list_iterations.append(iterations)
            break
        if not(Check_Generators_Limits()):
            print("Convergence has been reached")
            break
    T_bucle_out = time() - T_bucle_in
    print("\nLoop Time:", T_bucle_out," sg")
    tita_degree = np.rad2deg(tita)
    if not(divergence):
        Print_Voltage_Profile()
        Power_balance()
        write_results_on_files()
        return V_complex_profile