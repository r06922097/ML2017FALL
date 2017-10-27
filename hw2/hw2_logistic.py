import pandas as pd
import numpy as np
import math
import sys
import csv

input_path = sys.argv[5]
output_path = sys.argv[6]

# load testing data
df_test = pd.read_csv(input_path,encoding='big5')
df_test.columns = range(106)
x_test = np.array(df_test)

x_test = np.delete(x_test,1,1)
x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test), axis=1)

# load model
w = np.load('hw2_logistic.npy')

ans = []
for i in range(len(x_test)):
	ans.append([str(i+1)])
	a = np.dot(w,x_test[i])
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