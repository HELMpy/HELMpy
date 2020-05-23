
from os.path import basename 

import numpy as np
import pandas as pd



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
        Yshunt, Ytrans,
        branches_buses, Ybr_list,
        phase_barras, phase_dict,
    )


# Processing of .xlsx file data
def preprocess_case_data(
    algorithm, scale, 
    buses, N,
    branches, N_branches,
    generators, N_generators,
    N_coef, barras_CC=None,
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

    (   Yshunt, Ytrans,
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


# Separation of the voltage profile in its real and imaginary parts
#def Voltages_profile():
def separate_complex_to_real_imag_voltages(V_complex_profile):
    Vre = np.real(V_complex_profile)
    Vimag = np.imag(V_complex_profile)
    return Vre, Vimag


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
    return flag_violacion, Qg, Buses_type,


# Computing of the K factor for each PV bus and the slack bus.
# Only the PV buses are considered to calculate Pgen_total. The PV buses that were converted to PQ buses are NOT considered.
def compute_k_factor(Pg, Pg_sch, Pd, slack, N, list_gen):
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
    return K, Pg


# Set the slack's participation factor to 1 and the rest to 0. Classic slack bus model.
def K_slack_1(K,slack):
    K.fill(0)
    K[slack] = 1


# Computation of power flow trough branches and power balance
def power_balance(
    V_complex_profile, Ybr_list,
    N_branches, N, Shunt, slack, Pd, Qd, Pg, Qg,
    S_gen, S_load, S_mismatch, Q_limits, list_gen,
    Vre, Vimag, Yre, Yimag, branches_buses, algorithm,
    Pi=None, Qi=None, K=None, Pmismatch=None
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
            Vre, Vimag = separate_complex_to_real_imag_voltages(V_complex_profile)
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
