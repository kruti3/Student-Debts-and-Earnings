import read_data
import create_stats
import pickle
import sys
import numpy as np
import os.path

class StudentDebtEarning():
    """docstring for ClassName"""
    def __init__(self, arg):
        
        self.file_name = arg
        self.data_path = "../Data/" + self.file_name
        self.parameter_list = []
        self.data_array = []
        self.parameter_datatype = []
        self.numeric_data_array = np.array([])
        self.np_filename = "np_" + self.file_name[8:13] + ".txt"
        with open ('../Data/parameter_datatype', 'rb') as fp:
            self.parameter_datatype = pickle.load(fp)
        self.parameters = [parameter for parameter, datatype in self.parameter_datatype if datatype!=type('str')]
        
    def readData(self):

        self.parameter_list, self.data_array = read_data.getNpFromFile(self.data_path)
        
    def getDim(self):
        
        print "Number of columns", len(self.parameter_list)
        print "Number of rows", len(self.data_array)
        
    def convertToDataType(self):
     
        numeric_row = self.convertRowWiseToDataType(self.data_array[0])
        self.numeric_data_array = np.array(numeric_row)
        for one_row in self.data_array[1:]:
            numeric_row = self.convertRowWiseToDataType(one_row)
            self.numeric_data_array = np.vstack((self.numeric_data_array, numeric_row))

        np.savetxt("../Data/"+self.np_filename, self.numeric_data_array)
        
    def convertRowWiseToDataType(self, one_row):
        
        data_list = []
        element = 0
        index = 0
        for parameter, datatype in self.parameter_datatype:
            if datatype!=type('str'):
                data = one_row[index]
                if data == "PrivacySuppressed":
                        element = -1
                elif data == "NULL":
                        element = -2
                else:
                    element = datatype(data)
                data_list.append(element)
            index += 1
        
        return np.array(data_list)

    def createStats(self):

        self.numeric_data_array = np.genfromtxt("../Data/"+self.np_filename)        
        create_stats.writeToFile(self.numeric_data_array, self.np_filename, self.parameters)

def main(file_name):

    obj = StudentDebtEarning(file_name)
    
    if not os.path.isfile("../Data/"+obj.np_filename):
        obj.readData()
        obj.getDim()
        obj.convertToDataType()
    obj.createStats()
    pass


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print "Please enter file name"
        file = raw_input()
        main(file)