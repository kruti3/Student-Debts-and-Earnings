import numpy as np
import pickle

def eliminate(file_name, parameters_str, parameter_y):

    print "Eliminating columns with more than 30/100 of missing data"
    eliminated_columns = eliminate_columnwise("PS_" + file_name)
    eliminated_columns_1 = eliminate_columnwise("NULL_" + file_name)
    eliminated_columns.extend(eliminated_columns_1)
    eliminated_columns.sort()

    print "Eliminating rows with non-std values for Y output"
    eliminate_rows = eliminate_rowwise(file_name, parameters_str , parameter_y)
    eliminate_rows.sort()
    
    return eliminated_columns, eliminate_rows

def eliminate_columnwise(file_name):

    fp = open("../Data/"+file_name, "r+")
    stats = pickle.load(fp)
    
    ind = 0
    par_index_list = []
    for parameter, freq in stats:
        if freq>30.00:
            par_index_list.append(ind)
        ind += 1 
    return par_index_list

def eliminate_rowwise(file_name, parameters_str, parameter_y):
    
    data = np.genfromtxt("../Data/"+file_name)

    rows = data.shape[0]
    parameter_index = [parameters_str.index(x) for x in parameter_y]
    output_y = []
    eliminate_rows = []

    for i in range(rows):
        for index in parameter_index:
            output_y.append(data[i][index])
        if any(y==-1 or y==-2 for y in output_y):
            eliminate_rows.append(i)

        output_y = []
    
    return eliminate_rows