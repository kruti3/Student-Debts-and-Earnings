import os

for i in range(9):
    print i
    os.system("python project.py MERGED200" + str(i) +"_0" + str(i+1) + "_PP.csv")

print "09_10"
os.system("python project.py MERGED2009_10_PP.csv")

for i in range(10,15):
    print i
    os.system("python project.py MERGED20" + str(i) +"_" + str(i+1) + "_PP.csv")