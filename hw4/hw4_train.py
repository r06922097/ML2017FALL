import pandas as pd
import numpy as np
import sys
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, Dropout
from keras.layers import LSTM
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer 

# segmentation = open("segmentation.txt", "w", encoding = "utf-8")
# with open("sentence.txt", "r", encoding = "utf-8") as Corpus:
# 	for sentence in Corpus:
# 		sentence = sentence.strip("\n")
# 		pos = jieba.cut(sentence, cut_all = False)
# 		for term in pos:
# 			if term not in stopwordset:
# 				segmentation.write(term + " ")
# print("jieba 斷詞完畢，並已完成過濾停用詞!")
# segmentation.close()

# sentence = word2vec.Text8Corpus("sentence.txt")
# model = word2vec.Word2Vec(sentence, size = 50, window = 5, min_count = 4, workers = 4, sg = 1)
# model.wv.save_word2vec_format("sentence.model.bin", binary = True)
# print("model 已儲存完畢")

label_data = sys.argv[1]
unlabel_data = sys.argv[2]
word_vectors = KeyedVectors.load_word2vec_format('sentence.model.bin', binary=True)
vocab = dict([(k, v.index) for k, v in word_vectors.vocab.items()])

max_features = 20000
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

sentence = []
x_train = []
y_train = []

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
file = open(label_data,'r',encoding = 'utf8')
for line in file:
    sentence.append(line)

for line in sentence:
    tmp = line.replace('\n','')
    tmp2 = tmp.split(' +++$+++ ')
    y_train.append(int(tmp2[0]))
    x_train.append(tmp2[1])

#x_test = x_train[:len(x_train)//10] #first 1/10 is testing set
#x_train = x_train[len(x_train)//10:] #last 9/10 is training set
#y_test = y_train[:len(y_train)//10] #first 1/10 is testing set
#y_train = y_train[len(y_train)//10:] #last 9/10 is training set

token = Tokenizer(num_words = 5000,filters='\t\n')
token.word_index = vocab
x_train = token.texts_to_sequences(x_train)  
#x_test = token.texts_to_sequences(x_test)
print(len(x_train), 'train sequences')
#print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(64))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=3)
         # validation_data=(x_test, y_test))
#score, acc = model.evaluate(x_test, y_test,
#                            batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)

model.save('model_word2vec.h5')
