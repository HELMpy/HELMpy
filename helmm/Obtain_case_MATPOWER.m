% Script to obtain a base case to use as input for HELMpy
clear all; clc;

define_constants;
mpc = loadcase(case9); % Select the case of interest

file = 'Case.xlsx'; % Put a name to the .xlsx output file

buses = mpc.bus;
branches = mpc.branch;
generators = mpc.gen;

% warning( 'off', 'MATLAB:xlswrite:AddSheet' ) ;
xlswrite(file,buses,'Buses');
xlswrite(file,branches,'Branches');
xlswrite(file,generators,'Generators');