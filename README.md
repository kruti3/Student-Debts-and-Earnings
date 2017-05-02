# Student-Debts-and-Earnings


How to run the code?

1. Assigning data types for the raw data:

		python create_parameter_list.py

		Required files: parameter_names.txt
		File generated: parameter_datatype (List of tuples of parameter and datatype), 
						parameters 

2. Data preprocessing, Training, Prediction, Evaluation:

		python project.py *file name*

		file name is MERGEDyyyy_(yy+1)_PP.csv
		sample output: project_output.txt

		Files generated: Images: NULL_np_yy_yy+1.png, PS_np_yy_yy+1.png
						 DataFiles: np_yy_yy+1.txt, NULL_np_yy_yy+1.txt, PS_np_yy_yy+1.txt,
						 			Final_np_yy_yy+1.txt, Intermediate_np_yy_yy+1.txt

3. Running Experiment 1: (SVR, KNN)

		python experiment_1.py *file name*

		file name is MERGEDyyyy_(yy+1)_PP.csv
		sample output: exp_1_output.txt
		

4. Running Experiment 2: (Median Imputation)

		python experiment_2.py *file name* *median/mean*

		file name is MERGEDyyyy_(yy+1)_PP.csv
		sample output: exp_2_output.txt

5. Running Experiment 3: Applying PCA

		python experiment_3.py *file name*

		file name is MERGEDyyyy_(yy+1)_PP.csv
		sample output: exp_3_output.txt
