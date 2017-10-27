import csv
import numpy as np
import pandas as pd
import math
import sys

input_path = sys.argv[5]
output_path = sys.argv[6]

# load model
w = np.load('hw2_generative_w.npy')
b = np.load('hw2_generative_b.npy')

df_test = pd.read_csv(input_path,encoding='big5')
df_test.columns = range(106)
x_test = np.array(df_test)

ans = []
for i in range(len(x_test)):
	ans.append([str(i+1)])
	a = np.dot(w,x_test[i]) + b
	a = 1/(1+np.exp(-a))
	if a > 0.5:
		ans[i].append(1)
	elif a <= 0.5:
		ans[i].append(0)

filename = output_path
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()