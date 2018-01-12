import csv
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn import cluster
from keras.models import Model
from keras.layers import Input, Dense

image_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]


def load_test_file(test_path):
    df_load = pd.read_csv(test_path,encoding='big5')    
    image1_index = [int(i) for i in df_load['image1_index']]    
    image2_index = [int(i) for i in df_load['image2_index']]       
    return image1_index, image2_index

data = np.load(image_path)

data = data.astype('float32') / 255.

autoencoder = load_model('autoencoder_96402.h5')
encoder = load_model('encoder_96402.h5')
decoder = load_model('decoder_96402.h5')

encoded_imgs = encoder.predict(data)
decoded_imgs = decoder.predict(encoded_imgs)

clf = cluster.KMeans(init='k-means++', n_clusters=2).fit(encoded_imgs)


[image1_index,image2_index] = load_test_file(test_path)


ans = []
for i in range(len(image1_index)):
    ans.append([str(i)])
    if clf.labels_[image1_index[i]] == clf.labels_[image2_index[i]]:
        ans[i].append(1)
    else:
        ans[i].append(0)

filename = output_path
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["ID","Ans"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()
