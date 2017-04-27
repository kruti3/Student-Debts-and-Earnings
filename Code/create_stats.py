import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path


def writeToFile(data_array, file_name, parameters_list):

    if not os.path.isfile("../Data/PS_"+file_name):
        fp = open("../Data/PS_"+file_name, "w+")
        stats = get_count(data_array, -1, parameters_list)
        pickle.dump(stats, fp)
        fp.close()
    else:
        fp = open("../Data/PS_"+file_name, "r+")
        stats = pickle.load(fp)
        plot_line_graph(stats, "PS"+file_name[3:])
    
    if not os.path.isfile("../Data/NULL_"+file_name):
        stats = get_count(data_array, -2, parameters_list)
        fp = open("../Data/NULL_"+file_name, "w+")
        pickle.dump(stats, fp)
        fp.close()  
    else:
        fp = open("../Data/NULL_"+file_name, "r+")
        stats = pickle.load(fp)
        plot_line_graph(stats, "NULL"+file_name[3:])
    

def get_count(data_array, val, parameters_list):

    '''
    rows = data_array.shape[0]
    list_val = [val]
    
    for i in range(rows-1):
        total = data_array[i][data_array[i,:]==val]
        list_val.append(len(total))
    '''
    #print data_array.shape
    #print len(parameters_list)

    columns = data_array.shape[1]
    rows = data_array.shape[0]
    list_val = []
    sum_total = 0 
    
    for i in range(columns):
        total = data_array[data_array[:,i]==val, i]
        list_val.append((parameters_list[i],len(total)))
        sum_total += len(total)
    #list_val.append(sum_total)
    i = 0
    #print len(list_val)
    for parameter, freq in list_val:
        list_val[i] = (parameter, 100.0*freq/rows)
        i += 1

    if val==-1:
        print "PrivacySupressed: ", sum_total
    else:
        print "NULL: ", sum_total
    print list_val
    return list_val

def plot_line_graph(data_array, file_name):

        # Create line graphs for plotting cluster quality metric V/S Number of clusters
        range_x = [i for i in range(len(data_array))]
        percentage = [percent for parameter,percent in data_array]
        # Plot a line graph
        plt.figure(2, figsize=(6, 4))
        plt.scatter(range_x, percentage)
    
        # This plots the data
        plt.grid(True)
        plt.ylabel("Error")
        plt.xlabel("number")
        plt.title("freq")
        plt.xlim(0, len(data_array))                                                                                                                                                                                                                      
        plt.ylim(0, 100)

        plt.savefig(file_name[:-4])
        plt.close()