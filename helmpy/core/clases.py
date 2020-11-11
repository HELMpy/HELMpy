
from os.path import basename 

import numpy as np
import pandas as pd



# Branches data processing to construct Ytrans, Yshunt, branches_buses and others
def process_branches(branches, N_branches, case):
    for i in range(N_branches):
        FromBus = branches[0][i]
        ToBus = branches[1][i]
        R = branches[2][i]
        X = branches[3][i]
        BTotal = branches[4][i]
        Tap = branches[8][i]
        Shift_degree = branches[9][i]

        FB = case.Number_bus[FromBus] 
        TB = case.Number_bus[ToBus]
        case.Ybr_list.append([FB, TB, np.zeros((2,2), dtype=np.complex128)])
        Z = R + 1j*X
        if Tap == 0 or Tap == 1:
            if Z != 0:
                Yseries_ft = 1/Z
                if Shift_degree == 0:
                    case.Ybr_list[i][2][0,1] = case.Ybr_list[i][2][1,0] = -Yseries_ft
                else:
                    Shift = np.deg2rad(Shift_degree)
                    Yseries_ft_shift = Yseries_ft/(np.exp(-1j*Shift))
                    Yseries_tf_shift = Yseries_ft/(np.exp(1j*Shift))
                    case.Ybr_list[i][2][0,1] = -Yseries_ft_shift
                    case.Ybr_list[i][2][1,0] = -Yseries_tf_shift
                    if case.phase_barras[FB]:
                        if TB in case.phase_dict[FB][0]:
                            case.phase_dict[FB][1][ case.phase_dict[FB][0].index(TB) ] += Yseries_ft - Yseries_ft_shift
                        else:
                            case.phase_dict[FB][0].append(TB)
                            case.phase_dict[FB][1].append(Yseries_ft - Yseries_ft_shift)
                    else:
                        case.phase_dict[FB] = [[TB],[Yseries_ft - Yseries_ft_shift]]
                        case.phase_barras[FB] = True
                    if(case.phase_barras[TB]):
                        if( FB in case.phase_dict[TB][0]):
                            case.phase_dict[TB][1][ case.phase_dict[TB][0].index(FB) ] += Yseries_ft - Yseries_tf_shift
                        else:
                            case.phase_dict[FB][0].append(FB)
                            case.phase_dict[FB][1].append(Yseries_ft - Yseries_tf_shift)
                    else:
                        case.phase_dict[TB] = [[FB],[Yseries_ft - Yseries_tf_shift]]
                        case.phase_barras[TB] = True
                case.Ytrans[FB][TB] += -Yseries_ft
                case.Ytrans[FB][FB] +=  Yseries_ft
                case.Ytrans[TB][FB] += -Yseries_ft
                case.Ytrans[TB][TB] +=  Yseries_ft
            else:
                case.Ybr_list[i][2][0,1] = case.Ybr_list[i][2][1,0] = Yseries_ft = 0

            Bshunt_ft = 1j*BTotal/2
            case.Ybr_list[i][2][0,0] = case.Ybr_list[i][2][1,1] = Bshunt_ft + Yseries_ft
            case.Yshunt[FB] +=  Bshunt_ft
            case.Yshunt[TB] +=  Bshunt_ft
        else:
            Tap_inv = 1/Tap
            if Z != 0:
                Yseries_no_tap = 1/Z
                Yseries_ft = Yseries_no_tap * Tap_inv
                if Shift_degree == 0:
                    case.Ybr_list[i][2][0,1] = case.Ybr_list[i][2][1,0] = -Yseries_ft
                else:
                    Shift = np.deg2rad(Shift_degree)                
                    Yseries_ft_shift = Yseries_ft/(np.exp(-1j*Shift))
                    Yseries_tf_shift = Yseries_ft/(np.exp(1j*Shift))
                    case.Ybr_list[i][2][0,1] = -Yseries_ft_shift
                    case.Ybr_list[i][2][1,0] = -Yseries_tf_shift
                    if case.phase_barras[FB]:
                        if TB in case.phase_dict[FB][0]:
                            case.phase_dict[FB][1][ case.phase_dict[FB][0].index(TB) ] += Yseries_ft - Yseries_ft_shift
                        else:
                            case.phase_dict[FB][0].append(TB)
                            case.phase_dict[FB][1].append(Yseries_ft - Yseries_ft_shift)
                    else:
                        case.phase_dict[FB] = [[TB],[Yseries_ft - Yseries_ft_shift]]
                        case.phase_barras[FB] = True
                    if case.phase_barras[TB]:
                        if FB in case.phase_dict[TB][0]:
                            case.phase_dict[TB][1][ case.phase_dict[TB][0].index(FB) ] += Yseries_ft - Yseries_tf_shift
                        else:
                            case.phase_dict[FB][0].append(FB)
                            case.phase_dict[FB][1].append(Yseries_ft - Yseries_tf_shift)
                    else:
                        case.phase_dict[TB] = [[FB],[Yseries_ft - Yseries_tf_shift]]
                        case.phase_barras[TB] = True
                case.Ytrans[FB][TB] += -Yseries_ft
                case.Ytrans[FB][FB] +=  Yseries_ft
                case.Ytrans[TB][FB] += -Yseries_ft
                case.Ytrans[TB][TB] +=  Yseries_ft 
            else:
                case.Ybr_list[i][2][0,1] = case.Ybr_list[i][2][1,0] = Yseries_no_tap = Yseries_ft = 0
            
            B = 1j*BTotal/2
            Bshunt_f = (Yseries_no_tap + B)*(Tap_inv*Tap_inv) 
            Bshunt_t = Yseries_no_tap + B
            case.Ybr_list[i][2][0,0] = Bshunt_f
            case.Ybr_list[i][2][1,1] = Bshunt_t
            case.Yshunt[FB] +=  Bshunt_f - Yseries_ft
            case.Yshunt[TB] +=  Bshunt_t - Yseries_ft

        if TB not in case.branches_buses[FB]:
            case.branches_buses[FB].append(TB)
        if FB not in case.branches_buses[TB]:
            case.branches_buses[TB].append(FB)


def create_case_data_object_from_xlsx(grid_data_file_path, case_name=None):
    if not (case_name is None or type(case_name) is str):
        print("Erroneous argument type.")
        return None

    if case_name is None:
        # Extract file path and .xlsx termination to assign case_name
        case_name = basename(grid_data_file_path[0:-5])

    buses = pd.read_excel(grid_data_file_path, sheet_name='Buses', header=None)
    branches = pd.read_excel(grid_data_file_path, sheet_name='Branches', header=None)
    generators = pd.read_excel(grid_data_file_path, sheet_name='Generators', header=None)
    N = len(buses.index)
    N_generators = len(generators.index)
    N_branches = len(branches.index)

    # Create CaseData object
    case = CaseData(case_name, N, N_generators)

    case.N_branches = N_branches
    case.Pd[:] = buses[2]/100
    case.Qd[:] = buses[3]/100
    case.Shunt[:] = buses[5]*1j/100 + buses[4]/100
    case.Yshunt[:] =  np.copy(case.Shunt)

    for i in range(N):
        case.Number_bus[buses[0][i]] = i
        if buses[1][i] == 3:
            case.slack_bus = buses[0][i]
            case.slack = i

    pos = 0
    for i in range(N_generators):
        bus_i = case.Number_bus[generators[0][i]]
        if bus_i != case.slack:
            case.list_gen[pos] = bus_i
            pos += 1
        case.Buses_type[bus_i] = 'PVLIM'
        case.V[bus_i] = generators[5][i]
        case.Pg[bus_i] = generators[1][i]/100
        case.Qgmax[bus_i] = generators[3][i]/100
        case.Qgmin[bus_i] = generators[4][i]/100

    case.Buses_type[case.slack] = 'Slack'
    case.Pg[case.slack] = 0

    process_branches(branches, N_branches, case) 

    for i in range(N):
        case.branches_buses[i].sort()    # Variable that saves the branches

    case.Y[:] = np.copy(case.Ytrans)
    for i in range(N):
        if case.Yshunt[i].real != 0:
            case.conduc_buses[i] = True
        case.Y[i,i] += case.Yshunt[i]
        if case.phase_barras[i]:
            for k in range(len(case.phase_dict[i][0])):
                case.Y[i, case.phase_dict[i][0][k]] += case.phase_dict[i][1][k]

    return case


class CaseData:
    def __init__(self, name, N, N_generators):
        # case name
        self.name = name

        # case data
        self.N = N
        self.N_branches = np.int()
        self.slack_bus = np.int()
        self.slack = np.int()
        self.Number_bus = dict()
        self.Buses_type = ['PQ' for i in range(N)]
        self.list_gen = np.empty(N_generators-1, dtype=int)
        self.V = np.empty(N, dtype=np.float64)
        self.Pd = np.empty(N, dtype=np.float64)
        self.Qd = np.empty(N, dtype=np.float64)
        self.Pg = np.zeros(N, dtype=np.float64)
        self.Qgmax = np.empty(N, dtype=np.float64)
        self.Qgmin = np.empty(N, dtype=np.float64)
        self.Shunt = np.empty(N, dtype=np.complex128)
        self.conduc_buses = np.full(N, False)
        self.Yshunt =  np.empty(N, dtype=np.complex128)
        self.Ytrans = np.zeros((N,N), dtype=np.complex128)
        self.Y = np.zeros((N,N), dtype=np.complex128)
        self.Yre = np.real(self.Y)
        self.Yimag = np.imag(self.Y)
        self.branches_buses = [[i] for i in range(N)]
        self.Ybr_list = list()
        self.phase_barras = np.full(N, False)
        self.phase_dict = dict()

        # case parameters
        self.scale = 1

    def set_scale(self, scale):
        self.scale = scale
        self.Pd *= scale
        self.Qd *= scale
        self.Pg *= scale

    def reset_scale(self):
        self.Pd /= self.scale
        self.Qd /= self.scale
        self.Pg /= self.scale
        self.scale = 1


class RunVariables:
    def __init__(self, case, pv_bus_model, DSB_model, DSB_model_method, max_coef):
        # For readability
        N = case.N

        # Set number of coefficientis to start arrays. This is to to reduce the array size
        set_coef = 40 if max_coef > 40 else max_coef
        self.not_expanded = True # Variable execute expand_coef_arrays only once

        # Variables copied from case
        self.list_gen = np.copy(case.list_gen)
        self.Pg = np.copy(case.Pg)
        self.Buses_type = np.copy(case.Buses_type)
        self.N = N
        self.slack = case.slack

        # Length. Number of equations
        length = 2*N if DSB_model_method is None else 2*N+1
        self.length = length

        # Variables
        # Y_trans_mod is not included, but it may be.
        self.pv_bus_model = pv_bus_model
        self.DSB_model = DSB_model
        self.DSB_model_method = DSB_model_method
        self.max_coef = max_coef
        self.Qg = np.zeros(N, dtype=np.float64)
        self.V_complex_profile = np.empty(N, dtype=np.complex128)
        self.Vre = np.real(self.V_complex_profile)
        self.Vimag = np.imag(self.V_complex_profile)
        self.list_gen_remove = []
        self.list_coef = []
        self.solve = None
        self.coefficients = np.empty((length,set_coef), dtype=np.float64)
        self.Soluc_eval = np.empty((length,set_coef), dtype=np.float64)
        self.Soluc_no_eval = []
        self.V_complex = np.empty((N, set_coef), dtype=np.complex128)
        self.W = np.empty((N, set_coef), dtype=np.complex128)

        # HELM pv_bus_model 2
        if pv_bus_model == 2:
            self.barras_CC = dict()
            for i in self.list_gen:
                self.barras_CC[i] = np.empty(set_coef, dtype=np.complex128)
            self.VVanterior = np.empty(N, dtype=np.float64)
        else: self.barras_CC=None; self.VVanterior=None

        # HELM pv_bus_model 1
        if pv_bus_model == 1:
            self.Y_Vsp_PV = []
            self.Vre_PV = np.empty((N, set_coef), dtype=np.float64)
            self.resta_columnas_PV = np.empty(length, dtype=np.float64)
        else: self.Y_Vsp_PV=None; self.Vre_PV=None; self.resta_columnas_PV=None

        # Distributed slack
        self.K = np.zeros(N, dtype=np.float64)
        if DSB_model_method is not None:
            self.Pg_imbalance = np.sum(case.Pd) - np.sum(case.Pg)
        else: self.Pg_imbalance=None

        # DSB_model_method 2
        if DSB_model_method == 2:
            if pv_bus_model == 2:
                self.barras_CC[case.slack] = np.empty(set_coef, dtype=np.complex128)
                self.slack_CC = None
            else: 
                self.slack_CC  = np.empty(set_coef, dtype=np.complex128)
        else:
            self.slack_CC = None
        
    def expand_coef_arrays(self):
        """
        Expand the arrays of coefficients to the maximum number of coefficents (max_coef).
        They were originally set to 40 coeffcients if max_coef was higher than 40. 
        This was done to reduce the initial array size.
        """
        if self.not_expanded:
            self.not_expanded = False
            N = self.N
            max_coef = self.max_coef
            length = self.length
            set_coef = 40

            # coefficients
            coefficients = self.coefficients
            self.coefficients = np.empty((length,max_coef), dtype=np.float64)
            self.coefficients[:,0:set_coef] = coefficients

            # Soluc_eval
            Soluc_eval = self.Soluc_eval
            self.Soluc_eval = np.empty((length,max_coef), dtype=np.float64)
            self.Soluc_eval[:,0:set_coef] = Soluc_eval

            # V_complex
            V_complex = self.V_complex
            self.V_complex = np.empty((N, max_coef), dtype=np.complex128)
            self.V_complex[:,0:set_coef] = V_complex

            # W
            W = self.W
            self.W = np.empty((N, max_coef), dtype=np.complex128)
            self.W[:,0:set_coef] = W

            if self.pv_bus_model == 1:
                # Vre_PV
                Vre_PV = self.Vre_PV
                self.Vre_PV = np.empty((N, max_coef), dtype=np.float64)
                self.Vre_PV[:,0:set_coef] = Vre_PV

            if self.pv_bus_model == 2:
                # barras_CC
                for i in self.list_gen:
                    barras_CC = self.barras_CC[i]
                    self.barras_CC[i] = np.empty(max_coef, dtype=np.complex128)
                    self.barras_CC[i][:set_coef] = barras_CC

            # DSB_model_method 2
            if self.DSB_model_method == 2:
                if self.pv_bus_model == 2:
                    barras_CC = self.barras_CC[self.slack]
                    self.barras_CC[self.slack] = np.empty(max_coef, dtype=np.complex128)
                    self.barras_CC[self.slack][:set_coef] = barras_CC
                else: 
                    slack_CC = self.slack_CC
                    self.slack_CC = np.empty(max_coef, dtype=np.complex128)
                    self.slack_CC[:set_coef] = slack_CC

