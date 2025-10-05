"""
This file is the main model for determining exoplanets
- It uses a transformer architecture to try and determine if a light flux belongs to an exoplanet or not
- It is based on KOI fluxes 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

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

# This function saves the model weights
def summarySave(finalLoss, model):
    print(f'\nModel Finished Training!\nFinal Loss = {round(finalLoss, 2)}')
    torch.save(model.state_dict(), "modelSave.pth")
    print('Model Saved!')

# This is the main training and evaluation loop
def train_model(model, trainLoader, testLoader, device, epochs=50, lr=0.001):
    # Compute class weights based on training labels
    y_train_labels = np.array(trainLoader.dataset.y)
    classes = np.unique(y_train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1 = 0.0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        for x, y in tqdm(trainLoader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yHat = model(x)
            loss = criterion(yHat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(trainLoader)
        print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

        # --- Evaluation ---
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for x, y in testLoader:
                x, y = x.to(device), y.to(device)
                yHat = model(x)
                probs = F.softmax(yHat, dim=1)
                preds = torch.argmax(yHat, dim=1)

                all_probs.extend(probs[:,1].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        tp = np.sum((all_preds == 1) & (all_labels == 1))

        print(f"Top positive probs in this epoch: {np.sort(all_probs)[-5:]}")
        print(f"True positives: {tp} / {np.sum(all_labels==1)}")
        print(f"Test Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("-"*50)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "bestModel.pth")
            print(f"New best F1: {best_f1:.4f} — model saved!\n")

    # Save final model
    print("Training complete!")
    torch.save(model.state_dict(), "modelSave.pth")
    print("Model saved!")

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
    trainDataset = FluxData(xTrain, yTrain)
    testDataset = FluxData(xTest, yTest)
    print('Datasets created...')

    # Creating dataloader object 
    batchSize = 16
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
    print('Dataloaders created...')

    # Creating Model
    model = Model(xTrain.shape[1], 2)
    model.to(device)
    print('Model created, starting training...')
    train_model(model, trainLoader, testLoader, device)

if __name__ == "__main__":
    main()
