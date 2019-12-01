import os
import time
import pandas as pd
from multiprocessing import Pool

start_time = time.time()

path = "./Data/processedData.csv"
df = pd.read_csv(path, delimiter=",")
df = df.set_index("Name")
# print(df)
# print(list(df.columns.values))

geneList = list(df.columns.values)

def run_process(gene):
    path = "python C:/Users/Lee/PycharmProjects/workspace/PyTorch/PyTorchFirstStep/TrainingCPU.py {}".format(str(gene))
    os.system(path)

if __name__ == '__main__':
    pool = Pool(processes=7)
    pool.map(run_process, geneList)

print(time.time() - start_time)