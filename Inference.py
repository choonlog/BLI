import torch
import torch.nn.init as init
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import itertools
from sympy.logic import SOPform

path = "./Data/processedData.csv"
df = pd.read_csv(path, delimiter=",")
df = df.set_index("Name")
# print(df)
# print(list(df.columns.values))

geneList = list(df.columns.values)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.allModel = torch.nn.Sequential(
            torch.nn.Linear(feature_num, 256), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.Linear(256, real_num)
        )

    def forward(self, x):
        self.y_pred = self.allModel(x)
        return self.y_pred

initial = ""
logic = ""
featureSelectedGeneDic = {}
# genes = ["Gata2"]
for gene in geneList:

    # Set the file path
    filePathCor = "./Data/startingNetworkParCor.txt"

    # Read correlation file
    linesCor = open(filePathCor).read().splitlines()

    path = "./CheckPoint/" + str(gene) + ".pt"
    checkpoint = torch.load(path)

    featureSelectedGeneDic[gene] = checkpoint['featureSelectedGeneDic']

    feature_num = len(checkpoint['featureSelectedGeneDic'])
    # print(feature_num)
    real_num = 1

    # our model
    model = Model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n = feature_num
    table = list(itertools.product([1, 0], repeat=n))
    after = [list(inner) for inner in table] # List 안에 있는 tuple을 list로 바꿔준다.

    tensor = torch.FloatTensor(after)
    print(gene, feature_num, featureSelectedGeneDic[gene])

    mintermTrue = []
    mintermFalse = []
    activityList = []
    for i in tensor:
        BooleanActivity = i.view(1, -1)
        activity = model(BooleanActivity).data.item()
        # print(activity)
        activityList.append(activity)
        if activity > 0.5:
            mintermTrue.append(list(map(int, i.tolist())))
            # print("True: ", activity)
        else:
            mintermFalse.append(list(map(int, i.tolist())))
            # print("False: ", activity)

    fig = plt.figure()
    plt.title(gene)
    plt.xlabel('Activity')
    plt.ylabel('Frequency')

    plt.hist(activityList, 50, density=True, facecolor='g', alpha=0.75)

    plt.show()
    fig.savefig("./acitivityHisto/" + str(gene) + ".png")

    # print("mintermTrue: ", mintermTrue)
    # print("mintermFalse: ", mintermFalse)

    BooleanExp = SOPform(featureSelectedGeneDic[gene], mintermTrue, dontcares=None)
    initial += str(gene) + " = Random\n"
    logic += str(gene) + " *= " + str(BooleanExp) + "\n"

print(initial + "\n\n\n" + logic)