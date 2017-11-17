import numpy as np
import pandas as pd
import csv
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.models import load_model

training_data = sys.argv[1]

batch_size = 64
num_classes = 7
epochs = 60

df_load = pd.read_csv(training_data,encoding='big5')
y = np.array(df_load['label'])
y_train = np.zeros((len(y),num_classes))
for i in range(len(y)):
    if y[i] == 0:
        y_train[i][0] = 1
    elif y[i] == 1:
        y_train[i][1] = 1
    elif y[i] == 2:
        y_train[i][2] = 1
    elif y[i] == 3:
        y_train[i][3] = 1
    elif y[i] == 4:
        y_train[i][4] = 1
    elif y[i] == 5:
        y_train[i][5] = 1
    elif y[i] == 6:
        y_train[i][6] = 1

y_test = y_train[:len(y_train)//10] #first 1/10 is testing set
y_train = y_train[len(y_train)//10:] #last 9/10 is training set

x_train = []
for i in range(len(df_load)):
    tmp_list = df_load['feature'][i].split(' ')
    tmp_list2 = np.array([[int(i)] for i in tmp_list]) # convert every element i into [i]
    tmp_matrix = np.reshape(tmp_list2,(48,48,1))  #convert into 48*48 dimension
    x_train.append(tmp_matrix)
x_train = np.array(x_train)
x_test = x_train[:len(x_train)//10] #first 1/10 is testing set
x_train = x_train[len(x_train)//10:] #last 9/10 is training set



model = Sequential()
# Block 1
model.add(Conv2D(64, (5, 5), padding='same',input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2,2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Conv2D(128, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Conv2D(256, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))


model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, 
          batch_size=batch_size, 
          epochs=epochs,
          validation_data=(x_test, y_test), 
          shuffle=True)

# # Save model and weights
model.save('my_model_TA.h5')

# model = load_model('my_model.h5')
# df_test = pd.read_csv('test.csv',encoding='big5')

# x_test = np.array(df_test)
# x_test = []
# for i in range(len(df_test)):
#     tmp_list = df_test['feature'][i].split(' ')
#     tmp_list2 = np.array([[int(i)] for i in tmp_list]) # convert every element i into [i]
#     tmp_matrix = np.reshape(tmp_list2,(48,48,1))  #convert into 48*48 dimension
#     x_test.append(tmp_matrix)
# x_test = np.array(x_test)
# result = model.predict(x_test)

# ans = []
# result = result.tolist()
# for i in range(len(x_test)):
#     ans.append([str(i)])
#     ans[i].append(result[i].index(max(result[i])))

# filename = "predict_vggA.csv"
# text = open(filename, "w+")
# s = csv.writer(text,delimiter=',',lineterminator='\n')
# s.writerow(["id","label"])
# for i in range(len(ans)):
# 	s.writerow(ans[i]) 
# text.close()