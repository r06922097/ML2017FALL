import csv
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train = sys.argv[3]
Y_train = sys.argv[4]
input_path = sys.argv[5]
output_path = sys.argv[6]

# read in data
df_load = pd.read_csv(X_train,encoding='big5')
df_load.columns = range(106) #106 features
x = np.array(df_load)


df_load = pd.read_csv(Y_train,encoding='big5')
df_load.columns = range(1) #1 column
y = np.array(df_load)
y = y.ravel()


# xgboost train
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
param_dist = {'max_depth':20, 'eta':0.3, 'silent':1,'objective':'binary:logistic','n_estimators':200}
model = XGBClassifier()
model.fit(X_train, y_train)


# load testing data
df_test = pd.read_csv(input_path,encoding='big5')
df_test.columns = range(106)
X_test = np.array(df_test)


# make prediction
y_pred = model.predict(X_test)

ans = []
for i in range(len(X_test)):
	ans.append([str(i+1)])
	if y_pred[i] == 1:
		ans[i].append(1)
	else:
		ans[i].append(0)

filename = output_path
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()