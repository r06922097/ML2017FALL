from keras.models import load_model
import numpy as np
import pandas as pd
import csv
import sys

test_path = sys.argv[1]
output_path = sys.argv[2]

def load_test_file(test_path):
    df_load = pd.read_csv(test_path,encoding='big5')    
    UserID = [int(i) for i in df_load['UserID']]    
    MovieID = [int(i) for i in df_load['MovieID']]       
    return UserID, MovieID

[UserID,MovieID] = load_test_file(test_path)
UserID = np.array(UserID)
MovieID = np.array(MovieID)
UserBias = np.ones(len(UserID))
MovieBias = np.ones(len(MovieID))

model = load_model('model_train5_6.h5')
result = model.predict([MovieID,UserID,MovieBias,UserBias])

ans = []
for i in range(len(UserID)):
    ans.append([str(i+1)])
    ans[i].append(result[i][0])

filename = output_path
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["TestDataID","Rating"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()