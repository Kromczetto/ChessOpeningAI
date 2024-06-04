import pandas as pd
import numpy as np
openings_df = pd.read_csv('openings.csv')

move_columns = ['move1w', 'move1b', 'move2w', 'move2b', 'move3w', 'move3b', 'move4w', 'move4b']
openings_df['moves'] = openings_df[move_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
openings_df['OpeningEncoded'] = label_encoder.fit_transform(openings_df['Opening'])

all_moves = np.concatenate(openings_df['moves'].apply(lambda x: x.split()).values)
unique_moves = np.unique(all_moves)
move_encoder = {move: i+1 for i, move in enumerate(unique_moves)}  # +1 to reserve 0 for padding
openings_df['moves_encoded'] = openings_df['moves'].apply(lambda x: [move_encoder[move] for move in x.split()])
=======

#Reading data
data = pd.read_csv('openings.csv')

features = data[['Num Games', 'Perf Rating', 'Avg Player', 'Player Win %', 'Draw %','Moves']]
label = data['White_win%']

#Convert to numpy array
x = features.to_numpy()
y = label.to_numpy()


from tensorflow.keras.preprocessing.sequence import pad_sequences
max_length = max(openings_df['moves_encoded'].apply(len))
X = pad_sequences(openings_df['moves_encoded'], maxlen=max_length, padding='post')
y = openings_df['OpeningEncoded']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.neural_network import MLPClassifier

# Build and train the neural network
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
=======
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


from sklearn.metrics import accuracy_score

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

def predict_opening(moves):
    encoded_moves = [move_encoder[move] for move in moves.split()]
    padded_moves = pad_sequences([encoded_moves], maxlen=max_length, padding='post')
    prediction = mlp.predict(padded_moves)
    opening_name = label_encoder.inverse_transform(prediction)
    return opening_name[0]

# Example usage:
user_moves = "e4 e5 Nf3 Nc6"
print(predict_opening(user_moves))
