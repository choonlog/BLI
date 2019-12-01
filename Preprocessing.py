'''
1. qrtPCR data는 값이 낮을수록 activity가 높기 때문에 negation 시켜줘야 한다.
2. MinMax normalization이 필요하다.
3. Pesudotime trajectory 순서에 맞는 cell line을 골라내야 한다.

'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

path = "./Data/binary_expression_LMPP.txt"
with open(path) as file:
    genes = file.read().splitlines()[0]

with open(path) as file:
    cellLines = file.read().splitlines()[1:]

geneList = genes.split("\t")

cellLineList = []
for i in cellLines:
    cellLine = i.split("\t")
    cellLineList.append(cellLine[0])

# print(cellLineList)

'''
path = 'attractor_list_1cta_initCell.txt'
df = pd.read_csv(path, delimiter="\t")
df = df[geneList]
df.to_csv('attractor_list_1cta_initCell.csv')
'''

path = "./Data/data.csv"
df = pd.read_csv(path, delimiter=",")
df = df.set_index("Name")
df = df.loc[cellLineList]
df = df * -1
print(df)

# Column을 기준으로 한다.
data = df
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
data = scaler.fit_transform(data)
df = pd.DataFrame(data, columns=df.columns, index=list(df.index.values))

df.to_csv('./Data/processedData.csv')
print(df)


