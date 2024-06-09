import re
import pandas as pd
import numpy as np
def divide_string_into_array(text):
    lines = text.strip().split('\n')
    return lines

#PROBALY IT WILL NOT BE NESSESERY
def create_multi_dimensional_array(text):
    lines = text.strip().split('\n')
    multi_dim_array = []
    for line in lines:
        moves = line.split()
        multi_dim_array.append(moves)
    return multi_dim_array


data = pd.read_csv('openings.csv')
moves = data['Moves'].to_string(index=False)
moves = re.sub(r'\d+\.', '', moves)
arrayOfMoves = divide_string_into_array(moves)
arrayOfMoves = [move.strip() for move in arrayOfMoves]
#Testing moves value
# print(arrayOfMoves[0])

finalData = []
for i in range(len(data)):
    finalData.append({"evidence": arrayOfMoves[i], "label": data['Opening'].iloc[i]})

#Testing final data
#print(finalData[2])

