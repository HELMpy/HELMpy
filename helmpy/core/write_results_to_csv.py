from helmpy.util.root_path import ROOT_PATH


def write_results_to_csv(Mis, branch_data, case, bus_data, scale, algorithm):
    buses_csv_file_name = 'Results' + ' ' + \
                          algorithm + ' ' + \
                          str(case) + ' ' + \
                          str(scale) + ' ' + \
                          str(Mis) + ' ' + \
                          'buses' + '.csv'
    buses_csv_file_path = ROOT_PATH / 'data' / 'results' / buses_csv_file_name
    bus_data.to_csv(buses_csv_file_path)

    branches_csv_file_name = 'Results' + ' ' + \
                             algorithm + ' ' + \
                             str(case) + ' ' + \
                             str(scale) + ' ' + \
                             str(Mis) + ' ' + \
                             'branches' + '.csv'
    branches_csv_file_path = ROOT_PATH / 'data' / 'results' / branches_csv_file_name
    branch_data.to_csv(branches_csv_file_path)
