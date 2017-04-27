import csv
import numpy as np

def getNpFromFile(file_path):

	fp = open(file_path)
	reader = csv.reader(fp)

	data_array = []
	for row in reader:
		data_array.append(row)

	return data_array[0], data_array[1:]