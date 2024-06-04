import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


#Reading data
data = pd.read_csv('openings.csv')

features = data[['Num Games', 'Perf Rating', 'Avg Player', 'Player Win %', 'Draw %','Moves']]
label = data['White_win%']

#Convert to numpy array
x = features.to_numpy()
y = label.to_numpy()

#Preparing data to train and test model
traning = data[0:1500]
testing = data[1500:]

traning_features = traning[:,11]
traning_label = traning[:,1]
testing_features = testing[:,11]
testing_label = testing[:,1]

#Creating model
model = Sequential()
model.add(Input(shape=(traning_features.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#Compile model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

#Training model
model.fit(traning_features, traning_label, epochs=100, validation_data=(testing_features, testing_label))


print(traning_label)