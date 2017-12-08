import pandas as pd
import numpy as np
import csv
import sys
import wget
from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer 

word_vectors = KeyedVectors.load_word2vec_format('sentence.model.bin', binary=True)
vocab = dict([(k, v.index) for k, v in word_vectors.vocab.items()])

url = 'https://www.dropbox.com/s/wscph0jfxq7gh91/model_word2vec.h5?dl=1'
input_file = wget.download(url)
model = load_model(input_file)

input_path = sys.argv[1]
output_path = sys.argv[2]
test = []
max_features = 20000
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

sentence = []
x_test = []

file = open(input_path,'r',encoding = 'utf8')
for line in file:
    test.append(line)

test = test[1:len(test)]

for line in test:
    tmp = line.replace('\n','')
    tmp2 = tmp.split(',',1)
    x_test.append(tmp2[1])

token = Tokenizer(num_words = 5000,filters='\t\n')
token.word_index = vocab
x_test = token.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

result = model.predict(x_test)

ans = []
for i in range(len(x_test)):
	ans.append([str(i)])
	if result[i][0] > 0.5:
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
