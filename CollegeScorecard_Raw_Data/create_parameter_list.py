import re

pattern_datatype = { 'SAT.*':'float', 'MN_EARN_WNE_P.*':'int', 'MD_EARN_WNE_P.*':'int', '.*URL':'str', 'CC.*':'str', 'SAT.*':'float',
						'ACT.*':'float','COST.*':'int','TUITION.*':'int', 'NPT4.*':'int','UGDS_.*':'float','.*INC_PCT.*':'float',
						'RET_.*':'float', 'UG25.*':'float','PAR_ED_PCT_.*':'float','APPL_SCH_PCT_GE.*':'float','C150_.*':'float',
						'C200_.*':'float','.*_YR.*_RT':'float'
					}

with open('parameter_names.txt') as fp:
	content = fp.readlines()

parameter_list = [one.strip() for one in content]
count = 0

for one_pattern, datatype in pattern_datatype.items():
	found = filter(re.compile(one_pattern).match, parameter_list)
	if len(found):
		for one_found in found:
			ind = parameter_list.index(one_found)
			parameter_list[ind] = one_found + " " + datatype
			count += 1

print count		