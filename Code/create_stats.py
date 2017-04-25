import numpy as np
import pickle

def writeToFile(data_array, file_name):

    fp = open("STATS_"+file_name, "a+")
    stats = get_count(data_array, -1)
    pickle.dump(stats, fp)
    stats = get_count(data_array, -2)
    pickle.dump(stats, fp)
    fp.close()  


def get_count(data_array, val):

    rows = data_array.shape[0]
    list_val = [val]
    
    for i in range(rows-1):
        total = data_array[i][data_array[i,:]==val]
        list_val.append(len(total))

    sum_total = sum(list_val)
    list_val.append(sum_total)
    if val==-1:
        print "PrivacySupressed: ", sum_total
    else:
        print "NULL: ", sum_total
    return list_val