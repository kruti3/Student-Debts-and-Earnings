import re
import pickle

pattern_datatype = {  'SAT.*':float, 'MN_EARN_WNE_P.*':int, 'MD_EARN_WNE_P.*':int, '.*URL':str, 'CC.*':str, 'SAT.*':float,
						'ACT.*':float,'COST.*':int,'TUITION.*':int, 'NPT4.*':int,'UGDS_.*':float,'.*INC_PCT.*':float,
						'RET_.*':float, 'UG25.*':float,'PAR_ED_PCT_.*':float,'APPL_SCH_PCT_GE.*':float,'C150_.*':float,
						'C200_.*':float,'.*_YR.*_RT':float}

with open('parameter_list.txt') as fp:
	content = fp.readlines()

parameter_list = [one.strip() for one in content]
parameter_datatype = []
for i in range(len(parameter_list)):
	parameter_datatype.append((parameter_list[i],str))

count = 0
ind_list = set()
global_list = set([i for i in range(len(parameter_list))])

for one_pattern, boolean in pattern_datatype.items():
	found = filter(re.compile(one_pattern).match, parameter_list)
	if len(found):
		for one_found in found:
			ind = parameter_list.index(one_found)
			parameter_datatype[ind] = (one_found,boolean)
			count += 1
			ind_list.add(ind)

print global_list.difference(ind_list)
print count		

with open('parameter_datatype', 'wb') as fp:
    pickle.dump(parameter_datatype, fp)