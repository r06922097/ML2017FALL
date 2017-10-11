import pandas as pd
import numpy as np
import math
import sys
import csv

if __name__ == '__main__':

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	# read model
	w = np.load('model.npy')


	#testing data
	df_test = pd.read_csv(input_path,header=None,encoding='big5')
	df_test = df_test.replace('NR',0)

	x1_test = []

	for i in range(240):
		tmp_list = []
		#for j in range(18): ###for model 1
		j = 9
		#PM2.5 linear
		for k in range(2,11):
			tmp_list.append(float(df_test[k][j+i*18]))
		#PM2.5 quadratic
		for k in range(2,11):
			tmp_list.append(float(df_test[k][j+i*18])**2)
		j = 12
		#SO2 linear
		for k in range(2,11):
			tmp_list.append(float(df_test[k][j+i*18]))
		#SO2 quadratic
		for k in range(2,11):
			tmp_list.append(float(df_test[k][j+i*18])**2)
		j = 8
		#PM10 linear
		for k in range(2,11):
			tmp_list.append(float(df_test[k][j+i*18]))
		#PM10 quadratic
		for k in range(2,11):
			tmp_list.append(float(df_test[k][j+i*18])**2)
		x1_test.append(tmp_list)

	x_test = np.array(x1_test)
	x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test), axis=1)

	
	ans = []
	for i in range(len(x_test)):
		ans.append(["id_"+str(i)])
		a = np.dot(w,x_test[i])
		ans[i].append(a)

	filename = output_path
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
		s.writerow(ans[i]) 
	text.close()
