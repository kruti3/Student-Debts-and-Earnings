

def writeToFile(parameter_list, data_array, file_name):

	stats = get_count(data_array)

	fp = open("STATS_"+file_name, "w+")
	fp.write(stats)
	fp.close()	


def get_count(data_array):

	data_array = 
	