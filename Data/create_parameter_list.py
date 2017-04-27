import re
import pickle

pattern_datatype = {  'SAT.*':float, 'M[ND]_EARN_WNE_P6':int, 'SD_EARN_WNE_P6':float, '.*URL':str, 'CC.*':str, 'SAT.*':float,
						'ACT.*':float,'COSTT.*':int,'TUITIONFEE_.*':int, 'NPT4.*':int,
						#'/^UG$/':float, '/^UG_.*':int,'/^UGDS_.*':float,
						'UG':float,
						'RET_.*':float, '/^UG25ABV$/':float, 'PCTFLOAN':float, 'PCTPELL':float,
						#'PAR_ED_PCT_.*':float,'APPL_SCH_PCT_GE.*':float,
						'C\d{3}_4':float,'C\d{3}_L4':float, '.*_YR.*_RT':float, '.*_YR.*_N':float, '.*CIP.*':int,'MAIN':bool,
						'NUMBRANCH':int, 'HIGHDEG':int, 'ICLEVEL': int, 'SCH_DEG':int, 'ADM_RATE.*':float, 'PPTUG_.*':float,
						'DEBT_MDN':float, 'TUITFTE':float,
 						'INEXPFTE':float, 'AVGFACSAL':float, 'NUM4_.*':int,'NUM4\d_.*':int,'PFTFAC':float,'.*_PCT_.*':float,
 						'.*FAMINC.*':int,'PFTFTUG1_EF':float
 
						}

with open('parameter_names.txt') as fp:
	content = fp.readlines()

parameter_list = [one.strip() for one in content]
parameter_datatype = []
for i in range(len(parameter_list)):
	parameter_datatype.append((parameter_list[i],str))
count = 0
global_list = [i for i in parameter_list]

for one_pattern, boolean in pattern_datatype.items():
	found = filter(re.compile(one_pattern).match, parameter_list)
	if len(found):
		for one_found in found:
			ind = parameter_list.index(one_found)
			parameter_datatype[ind] = (one_found, boolean)
			global_list.remove(one_found)
			count += 1

print global_list
print count		

with open('parameter_datatype', 'wb') as fp:
    pickle.dump(parameter_datatype, fp)