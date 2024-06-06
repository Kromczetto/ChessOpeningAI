import re

def divide_string_into_array(text):
    lines = text.strip().split('\n')
    return lines

def create_multi_dimensional_array(text):
    lines = text.strip().split('\n')
    multi_dim_array = []
    for line in lines:
        moves = line.split()
        multi_dim_array.append(moves)
    return multi_dim_array

partie = """
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.Bc4
1.e4 Nf6 2.e5 Ng8
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.c4 Nb6 5.exd6
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.c4 Nb6 5.f4
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.c4 Nb6 5.f4 g6
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.c4 Nb6 5.f4 dxe5 6.fxe5 Nc6 7.Be3
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.c4 Nb6 5.f4 Bf5
1.e4 Nf6
1.e4 Nf6 2.e5 Nd5 3.c4 Nb6 4.c5 Nd5 5.Nc3 e6 6.Bc4
1.e4 Nf6 2.d3
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.Nf3
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.Nf3 g6
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.Nf3 Bg4 5.c4 Nb6 6.Be2
1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.Nf3 g6 5.Bc4 Nb6 6.Bb3 Bg7 7.a4
"""

tab= re.sub(r'\d+\.', '', partie)
tablica_partii = divide_string_into_array(tab)
tablica_partii_wielowymiarowa = create_multi_dimensional_array(tab)


print(tablica_partii_wielowymiarowa[1][3])
