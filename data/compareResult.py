import random		
import csv
import platform
import numpy

file1 = "vis_output_test_cpu.csv"
file2 = "vis_output_test.csv"

diff = 0.0

with open(file1) as f1, open(file2) as f2:
    f1_reader = csv.reader(f1)
    f2_reader = csv.reader(f2)
    header = next(f1_reader)
    header = next(f2_reader)
    for row_f1 in f1_reader:
        row_f2 = next(f2_reader)
        d_1 = numpy.fromstring(row_f1[0], dtype=float, sep=' ')
        d_2 = numpy.fromstring(row_f2[0], dtype=float, sep=' ')
        for i in range (6):
            dif = numpy.sqrt((d_2[i] - d_1[i])**2)
            if(dif > 0.001):
                print(dif)
            diff += dif
            
print(diff)
      


