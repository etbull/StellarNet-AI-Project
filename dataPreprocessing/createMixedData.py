"""
This file creates the mixes data set (mixed.csv)
- Contains test data from noth exoTrain and amaTrain
"""

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        testDF.to_csv(r'backend\\data\\mixed.csv', index=False)
        print('Saved to CSV!')
        exit()
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

if __name__ == "__main__":
    main()