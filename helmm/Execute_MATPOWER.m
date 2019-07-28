% Script to perform power flow of a desired case in matpower and to obtain
% results in .xlsx and .txt files
clear all; clc;

define_constants;
mpc = loadcase(case9); % Select the case of interest

% Modify initial seed and loadability scale of the case for all buses
Scale = 1;
mpc.bus(:, VM) = 1; % V magnitude
mpc.bus(:, VA) = 0; % V phase angle
mpc.bus(:, PD) = mpc.bus(:, PD)*Scale; % Active power demand
mpc.bus(:, QD) = mpc.bus(:, QD)*Scale; % Reactive power demand
mpc.gen(:, PG) = mpc.gen(:, PG)*Scale; % Active power generation

mpopt = mpoption('pf.nr.max_it',15, 'pf.enforce_q_lims',1,'pf.tol',10^-8, 'verbose',3, 'out.bus',1, 'out.branch',1, 'out.all',1, 'out.gen',1, 'out.lim.all',1); % Chose options for the PF, visit Matpower User's Manual for details
results = runpf(mpc, mpopt,'Results MATPOWER case.txt'); % Run the PF and put a name to the .txt output file

buses = results.bus;
branches = results.branch;
generators = results.gen;

file = 'Results MATPOWER case.xlsx'; % Put a name to the .xlsx output file
xlswrite(file,buses,'Buses');
xlswrite(file,branches,'Branches');
xlswrite(file,generators,'Generators');