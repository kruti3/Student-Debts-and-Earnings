import read_data
import create_stats
import sys
import numpy as np

class StudentDebtEarning():
	"""docstring for ClassName"""
	def __init__(self, arg):
		self.file_name = arg
		self.data_path = "../CollegeScorecard_Raw_Data/" + arg
		self.parameter_list = []
		self.data_array = []
	
	def readData(self):

		self.parameter_list, self.data_array = read_data.getNpFromFile(self.data_path)

	def getDim(self):
		
		print "Number of columns", len(self.parameter_list)
		print "Number of rows", len(self.data_array)
		i=0
		for one in self.parameter_list:

			print type(self.data_array[0][i])
			i+=1

	def createStats(self):
		create_stats.writeToFile(self.parameter_list, self.data_array, self.file_name)

def main(file_name):

	obj = StudentDebtEarning(file_name)
	obj.readData()
	obj.getDim()
	#obj.createStats()

	pass


if __name__ == '__main__':
	if len(sys.argv) == 2:
		main(sys.argv[1])
	else:
		print "Please enter file name"
		file = raw_input()
		main(file)