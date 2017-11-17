from __future__ import print_function
import keras
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

training_data = sys.argv[1]

batch_size = 64
num_classes = 7
epochs = 65

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


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, (5, 5), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2,2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('cifar10_model55.h5')