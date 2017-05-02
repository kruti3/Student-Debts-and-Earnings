import read_data
import create_stats
import refine_data
import pickle
import sys
import numpy as np
import os.path
import math
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from sklearn.cross_validation import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score

class StudentDebtEarning():

    """docstring for ClassName"""
    def __init__(self, arg):
        '''
        file_name: Input file name
        data_path: path for input file
        parameter_datatype : list of tuples of parameter,datatype
        numeric_data_array : stores np array of all the data
        parameter_y : strings for predictor variables
        parameter_str: strings for all the parameters in order
        eliminate_row/columns : indices of columns to be removed
        exclude_list : -1 is denoted for privacy suppressed/ -2 for NULL values
        '''
        self.file_name = arg
        self.data_path = "../Data/" + self.file_name
        self.parameter_list = []
        self.data_array = []
        self.parameter_datatype = []
        self.numeric_data_array = np.array([])
        self.parameter_y = ['MD_EARN_WNE_P6', 'DEBT_MDN', 'RPY_5YR_RT']
        self.np_filename = "np_" + self.file_name[8:13] + ".txt"
        with open ('../Data/parameter_datatype', 'rb') as fp:
            self.parameter_datatype = pickle.load(fp)
        self.parameters_str = [parameter for parameter, datatype in self.parameter_datatype if datatype!=type('str')]
        self.exclude_val_list = [-1, -2]
        self.eliminate_columns = []
        self.eliminate_rows = []
        self.X = []
        self.Y = []
        
    def readData(self):
        '''
        Read raw data into lists
        '''
        print "Reading Data from raw file", self.file_name
        self.parameter_list, self.data_array = read_data.getNpFromFile(self.data_path)
        
    def getDim(self):
        '''
        Get the dimenions of the raw data
        '''
        print "Number of columns", len(self.parameter_list)
        print "Number of rows", len(self.data_array)
        
    def convertToDataType(self):
        '''
        On the basis of parameter_datatype file, each of the columns are typecasted from str to respective
        datatypes and stored as numpy array and written to a file np_yy_yy+1.txt

        '''
        print "Typecasting the raw data to respective datatypes"
        numeric_row = self.convertRowWiseToDataType(self.data_array[0])
        self.numeric_data_array = np.array(numeric_row)
        for one_row in self.data_array[1:]:
            numeric_row = self.convertRowWiseToDataType(one_row)
            self.numeric_data_array = np.vstack((self.numeric_data_array, numeric_row))

        print "Obtained numpy array for the data"
        print self.numeric_data_array.shape    
        print "File saved as np_(yy)_(yy+1).txt"
        np.savetxt("../Data/"+self.np_filename, self.numeric_data_array)
        
    def convertRowWiseToDataType(self, one_row):
        '''
        Typcasting from str to respective datatype row wise
        '''
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

    def createStats(self, file_name):
        '''
        Create stats for NULL and PrivacySuppressed values percentages, create graphs
        as a percentage of total numer of schools
        
        '''
        self.numeric_data_array = np.genfromtxt("../Data/"+file_name)    
        
        print "Creating stats for NULL, PrivacySuppressed frequency"
        print self.numeric_data_array.shape    
        
        create_stats.writeToFile(self.numeric_data_array, file_name, self.parameters_str)
        pred_index_list = []

        for one in self.parameter_y:
            pred_index_list.append(self.parameters_str.index(one))
        
        print "Creating stats for Y(Dependent Variables) columns"
        print create_stats.get_count_y(self.numeric_data_array, pred_index_list,self.parameters_str, self.exclude_val_list)

    def get_column_by_name(self, string, file_name):

        '''
        get data such as max, min, count of null, count of privacy suppressed vals, mean of a column 
        by specifying the respective column name
        '''
        index = self.parameters_str.index(string)
        self.numeric_data_array = np.genfromtxt("../Data/"+file_name)    

        print "Get range of data of a column"
        column_vec = []
        count_1 = 0
        count_2 = 0

        for i in range(self.numeric_data_array.shape[0]):
            if (self.numeric_data_array[i][index])==-1:
                count_1 += 1
            elif (self.numeric_data_array[i][index])==-2:
                count_2 += 1
            elif int(self.numeric_data_array[i][index])!=(-1) or int(self.numeric_data_array[i][index])!=(-2):
                column_vec.append(self.numeric_data_array[i][index])
            
        return max(column_vec), min(column_vec), count_1, count_2, (sum(column_vec)*1.0)/self.numeric_data_array.shape[0]

    def refine_data_eliminate(self):
        '''
        Eliminate data if Y output is absent or Number of missing value percent is above 30%
        final_parameters file contains str of parameters present in the data
        Intermediate_** file contains filtered data
        '''
        self.eliminate_columns, self.eliminate_rows = refine_data.eliminate(self.np_filename, self.parameters_str, self.parameter_y)
        
        print self.numeric_data_array.shape    
        self.numeric_data_array = np.genfromtxt("../Data/"+self.np_filename)    
        self.numeric_data_array = np.delete(self.numeric_data_array, self.eliminate_columns, axis=1)
        self.numeric_data_array = np.delete(self.numeric_data_array, self.eliminate_rows, axis=0)
        
        eliminate_columns_reverse = self.eliminate_columns
        eliminate_columns_reverse.reverse()
        
        for index in eliminate_columns_reverse:
            del self.parameters_str[index]

        print "Saved list of final set of parameters after data elimination"
        fp = open("../Data/final_parameters", "w+")
        pickle.dump(self.parameters_str, fp)
        fp.close()

        print "Saved NP data after data elimination"
        np.savetxt("../Data/Intermediate_"+self.np_filename, self.numeric_data_array)
        print self.numeric_data_array.shape    

    def refine_data_impute(self, strategy_str):
        '''
        replacing non standard values

        '''
        self.numeric_data_array = np.genfromtxt("../Data/Intermediate_"+self.np_filename)    
        self.modify_NULL_values(-2)

        print "Imputing privacySuppressed values by the mean of each column"
        imp = Imputer(missing_values=-1,strategy=strategy_str)
        self.numeric_data_array = imp.fit_transform(self.numeric_data_array)
        

        print "Saved Data in Final_np_yy_(yy+1).txt after data imputation and elimination"
        print self.numeric_data_array.shape    
        np.savetxt("../Data/Final_"+self.np_filename, self.numeric_data_array)

    def modify_NULL_values(self, val):

        '''
        replacing NULL values by mean of respective columns
        '''
        print "Replacing NULL values by min of each column"
        min_values = np.amin(self.numeric_data_array, axis=0)
        columns = self.numeric_data_array.shape[1]

        for i in range(columns):
            self.numeric_data_array[self.numeric_data_array[:,i]==val,i] = min_values[i]

    def split(self, one_parameter):
        '''
        Creating X and Y from final_np_yy_yy+1.txt file
        where y is specified by the string one_parameter

        '''
        fp = open("../Data/final_parameters", "r+")
        self.parameters_str = pickle.load(fp)
        fp.close()

        self.numeric_data_array = np.genfromtxt("../Data/Final_"+self.np_filename)    
        ind = self.parameters_str.index(one_parameter)
        self.Y = self.numeric_data_array[:,ind]
        
        ind_del_list = []
        for one_parameter in self.parameter_y:
            ind_del_list.append(self.parameters_str.index(one_parameter))
        ind_del_list.sort()
        
        print "Deleting columns of Y variable from the dataset, obtaining Y vectors for debt and earnings"
        self.X = np.delete(self.numeric_data_array, ind_del_list, axis=1)

        self.X_train , self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)
                
    def feature_selection(self, regression_tuple, param_dict):

        '''
        regression tuple contains the regressor instance, param_dict contains hyperparametrs
        This is a pipeline created for selecting k best features and hyperparameter selection 
        '''
        number_of_features = self.X.shape[1]

        pipeline_list = [('sel', SelectKBest())]
        pipeline_list.append(regression_tuple)
        pipeline = Pipeline(pipeline_list)

        param = {'sel__k':[i for i in range(10,number_of_features+1,5)]}
        if len(param_dict):
            param.update(param_dict)
        
        print "Using Kfold = 10"
        cv = KFold(n=10)

        print "Applying GridSearch CV"
        self.regr = GridSearchCV(pipeline, param)

    def train(self):
        '''
        Fit data into the regressor
        '''
        self.regr.fit(self.X_train, self.Y_train)
        print self.regr.best_params_
        
    def predict(self):
        '''
        perform predictions
        '''
        self.Y_pred = self.regr.predict(self.X_test)
        
    def calculate_error(self):
        '''
        Calculate root mean squared error
        '''
        error = math.sqrt(mean_squared_error(self.Y_test, self.Y_pred))
        return error

    def calculate_r2_score(self):
        '''
        Calcualte r2 error
        '''
        error = r2_score(self.Y_test, self.Y_pred)
        return error

def main(file_name):

    obj = StudentDebtEarning(file_name)
    
    if not os.path.isfile("../Data/"+obj.np_filename):
        #Read data from raw file
        obj.readData()

        #Get dimension of the data
        obj.getDim()

        #Convert the read data into respective datatypes and assign to numpy and write to file np_yy_(yy+1).txt
        obj.convertToDataType()

    #Create stats for NULL and PrivacySuppressed values percentages, create graphs
    obj.createStats(obj.np_filename)
    
    #Eliminate data if Y output is absent or Number of missing value percent is above 30%
    obj.refine_data_eliminate()
    obj.refine_data_impute(sys.argv[2])
    
    #Get minimum and max value of each column (excluding PrivacySuppressed and NULL values)
    '''
    fp = open("../Data/PS_Intermediate_"+obj.np_filename, "r+")
    data_array = pickle.load(fp)
    print "why"
    for one,freq in data_array:
        print one,freq
        print obj.get_column_by_name(one, "Intermediate_"+obj.np_filename)
    
    '''

    # Different models applied
    regression_str = ["Linear regression", "Ridge", "Lasso", "Decision Tree"]
    regression_tuple = [('lr',linear_model.LinearRegression(normalize=True)), ('ridge', linear_model.Ridge(random_state=0)),
                             ('lasso', linear_model.Lasso(random_state=0)), ('tree',DecisionTreeRegressor())]
    regression_param = [{}, {'ridge__alpha':[0.1,1.0,10.0]}, 
                        {'lasso__alpha': [0.1,1.0,2.0,5.0,10.0]},
                        {'tree__max_depth': [5,10,15,20,25,30,50,100,150,200]}]
    
    
    # Predicting for each variable
    for one_parameter in ['MD_EARN_WNE_P6', 'DEBT_MDN', 'RPY_5YR_RT']:
        print "\n\nY :",one_parameter
        obj.split(one_parameter)
        i = 0

        # Predicting using each model
        for i in range(len(regression_str)):
            print "\nRegressor: ", regression_str[i]
            # Hyperparameter/ Feature selection/ CV
            obj.feature_selection(regression_tuple[i], regression_param[i])

            # Training and Prediction
            obj.train()
            obj.predict()

            #Calculating error coefficients
            root_mean_sq_error = obj.calculate_error()
            print "Root Mean Squared Error", root_mean_sq_error
            print "R2 error", obj.calculate_r2_score()
            a,b,c,d,mean_column = obj.get_column_by_name(one_parameter, "Final_"+obj.np_filename)
            print "Percent root mean squared error", (100.0*root_mean_sq_error)/mean_column

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        print "Please enter file name"
        file = raw_input()
        main(file)