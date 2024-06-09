import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Function to divide the string into array
def divide_string_into_array(text):
    lines = text.strip().split('\n')
    return lines

#Load data
data = pd.read_csv('openings.csv')
moves = data['Moves'].to_string(index=False)
moves = re.sub(r'\d+\.', '', moves)
arrayOfMoves = divide_string_into_array(moves)
arrayOfMoves = [move.strip() for move in arrayOfMoves]

#Prepare final data
finalData = []
for i in range(len(data)):
    finalData.append({"evidence": arrayOfMoves[i], "label": data['Opening'].iloc[i]})

#Split evidence and labels
evidence = [row["evidence"] for row in finalData]
labels = [row["label"] for row in finalData]

#Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(evidence)
x_sequences = tokenizer.texts_to_sequences(evidence)

#Pad sequences to ensure uniform length
max_sequence_length = max(len(seq) for seq in x_sequences)
x_padded = pad_sequences(x_sequences, maxlen=max_sequence_length)

#Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

#Split data into training and testing sets
x_training, x_testing, y_training, y_testing = train_test_split(
    x_padded, y_encoded, test_size=0.4
)

#Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(len(np.unique(y_encoded)), activation="softmax"))

#Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Train the model
model.fit(x_training, y_training, epochs=50, validation_data=(x_testing, y_testing))

#Evaluate the model
model.evaluate(x_testing, y_testing, verbose=2)

#Function which predict
def predict_opening(moves, tokenizer, model, label_encoder, max_sequence_length):
    moves = re.sub(r'\d+\.', '', moves)
    moves = moves.strip()
    sequence = tokenizer.texts_to_sequences([moves])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]


#Change new_moves to new chees move
new_moves = "d4 d5 c4 e5 dxe5 d4 Nf3 Nc6 Nbd2"
predicted_opening = predict_opening(new_moves, tokenizer, model, label_encoder, max_sequence_length)
print(f"The predicted opening for the moves '{new_moves}' is: {predicted_opening}")
