import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Reading data
data = pd.read_csv('openings.csv')

x = data.iloc[:,-1].values
y = data.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential([
    Dense(64, input_dim=x_train.shape[1], activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

#model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train model
model.fit(x_train,y_train, epochs=50, batch_size=10, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Zapisanie wytrenowanego modelu do pliku
model.save('trained_model.h5')

