"""
This collects inference data to be displayed on the website
- Puts it into JSON file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

"""
This model is based off a CNN architecture, and includes multiple convolution and batch normalisation layers
- The original model I had was a lot more complex, but lead to overfitting
"""
class Model(nn.Module):
    def __init__(self, input_length, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear((input_length // 2) * 128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
The Data class defines the dataset which returns the training and target values when called
"""
class FluxData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# This function adds the amature positive data to the training and testing data frames
def addAmature(trainDF, testDF, trainingPathAma, testingPathAma, status):
    if status:
        amaTrainDF = pd.read_csv(trainingPathAma)
        amaTestDF = pd.read_csv(testingPathAma)  
        amaTrainDF["LABEL"] = 2
        amaTestDF["LABEL"] = 2
        trainDF = pd.concat([trainDF, amaTrainDF], ignore_index=True)
        testDF = pd.concat([testDF, amaTestDF], ignore_index=True)
        print('Amature Data Sucessfully Intergrated...')
    else:
        print('Amature Data Not Intergrated...')
    return trainDF, testDF

# This function does data loading + standardization
def standardise(trainingPathExo, testingPathExo, trainingPathAma, testingPathAma):
    trainDF = pd.read_csv(trainingPathExo)
    testDF = pd.read_csv(testingPathExo)
    trainDF, testDF = addAmature(trainDF, testDF, trainingPathAma, testingPathAma, True)
    
    # Make labels 0/1
    trainDF['LABEL'] = trainDF['LABEL'] - 1
    testDF['LABEL'] = testDF['LABEL'] - 1

    scaler = StandardScaler()
    scaler.fit(trainDF.drop(columns=['LABEL']))
    xTrain = scaler.transform(trainDF.drop(columns=['LABEL']))
    xTest = scaler.transform(testDF.drop(columns=['LABEL']))
    yTrain = trainDF['LABEL'].values
    yTest = testDF['LABEL'].values
    print("Class distribution (train):", np.bincount(yTrain))
    print("Class distribution (test):", np.bincount(yTest))
    return xTrain, xTest, yTrain, yTest

# This function augments the positive exoplanets so that there's more data
def augmentPositives(X, y, factor=3, shiftMax=1500, noiseStd=0.001):
    xAug, yAug = X.copy(), y.copy()
    pos_idxs = np.where(y==1)[0]
    for _ in range(factor-1):
        for i in pos_idxs:
            sample = np.roll(X[i], np.random.randint(-shiftMax, shiftMax))
            sample += np.random.normal(0, noiseStd, size=sample.shape)
            xAug = np.vstack([xAug, sample])
            yAug = np.append(yAug, 1)
    return xAug, yAug

# This is the main training and evaluation loop
# We are collecting and putting into csv, id, flux data, Prediction, actual, pred value
def test_model(model, testLoader, device):
    model.eval()
    df = pd.DataFrame({'id':[], 'fluxData':[], 'predictedClass':[], 'actualClass':[], 'predCertainty':[]})
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()  # use class weights if needed

    with torch.no_grad():
        total_loss = 0
        count = 0
        rows = []
        for x, y in testLoader:
            x, y = x.to(device), y.to(device)
            yHat = model(x)
            loss = criterion(yHat, y)
            total_loss += loss.item()
            probs = torch.softmax(yHat, dim=1)
            preds = torch.argmax(probs, dim=1)
            for i in range(len(y)):
                rows.append({
                    'id': 1000 + count,
                    'fluxData': x[i].cpu().numpy().tolist(),  
                    'predictedClass': preds[i].item(),
                    'actualClass': y[i].item(),
                    'predCertainty': probs[i, preds[i]].item()
                })
                count += 1

        df = pd.DataFrame(rows)
        df.to_json("backend/data/inferenceData.json", orient="records", indent=2)
        print(f"Inference data saved! Total loss: {total_loss:.4f}")
        exit()

# The main loop of the program 
def main():
    print('Process started...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)

    trainingPathExo = r'backend/data/exoTrain.csv'
    testingPathExo = r'backend/data/exoTest.csv'
    trainingPathAma = r'backend/data/amaTrain.csv'
    testingPathAma = r'backend/data/amaTest.csv'

    # First, standardising data, function returns 4 lists, all standardised. Also augments data to add more positives
    xTrain, xTest, yTrain, yTest = standardise(trainingPathExo, testingPathExo, trainingPathAma, testingPathAma)
    xTrain, yTrain = augmentPositives(xTrain, yTrain, 3)

    # Next, creating dataset objects
    testDataset = FluxData(xTest, yTest)
    print('Datasets created...')

    # Creating dataloader object 
    batchSize = 16
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
    print('Dataloaders created...')

    # Creating Model
    model = Model(xTrain.shape[1], 2)
    model.load_state_dict(torch.load(r"backend\\modelMixed.pth"))
    model.to(device)

    print('Model created, starting testing...')
    test_model(model, testLoader, device)

if __name__ == "__main__":
    main()
