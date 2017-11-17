import numpy as np
import pandas as pd
import csv
import sys
from keras.models import load_model
from collections import Counter
import wget

testing_data = sys.argv[1]
output_file = sys.argv[2]

url = 'https://www.dropbox.com/s/kr3j31yn59jm6fj/40x40_vggA2.h5?dl=1'
url4 = 'https://www.dropbox.com/s/jjavck1k4iow7ly/my_model_TA.h5?dl=1'
url5 = 'https://www.dropbox.com/s/wg0pa7ervximpbl/train3.h5?dl=1'
input_file = wget.download(url)
input_file4 = wget.download(url4)
input_file5 = wget.download(url5)

model = load_model(input_file)
model4 = load_model(input_file4)
model5 = load_model(input_file5)
df_test = pd.read_csv(testing_data,encoding='big5')

x_test = []
for i in range(len(df_test)):
    tmp_list = df_test['feature'][i].split(' ')
    tmp_list2 = np.array([[int(i)] for i in tmp_list]) # convert every element i into [i]
    tmp_matrix = np.reshape(tmp_list2,(48,48,1))  #convert into 48*48 dimension
    x_test.append(tmp_matrix)
x_test = np.array(x_test)
result4 = model4.predict(x_test)
result5 = model5.predict(x_test)

x_test = []
for i in range(len(df_test)):
    tmp_list = df_test['feature'][i].split(' ')
    tmp_list2 = np.array([[int(i)] for i in tmp_list]) # convert every element i into [i]
    tmp_matrix = np.reshape(tmp_list2,(48,48,1))  #convert into 48*48 dimension
    for j in range(9):
        tmp_matrix2 = np.reshape(tmp_matrix[j:j+40,j:j+40],(40,40,1)) #augment to 9 40*40 dimension matrix
        x_test.append(tmp_matrix2)
x_test = np.array(x_test)
result = model.predict(x_test)

ans = []
result = result.tolist()
for i in range(len(df_test)):
    ans.append([str(i)])
    predict_list = result[9*i:9*i+9]
    predict_list.append(result4[i])
    predict_list.append(result5[i])
    predict_list = np.array(predict_list)
    predict_sum = predict_list.sum(0)
    ans[i].append(predict_sum.tolist().index(max(predict_sum)))

text = open(output_file, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()