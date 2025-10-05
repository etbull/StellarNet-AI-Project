"""
This file extracts the flux data from amature exoplanet observations
- It creates a csv where each row is a unique UID
- The light data column used is RELATIVE_FLUX_WITHOUT_SYSTEMATICS
"""

import pandas as pd
import glob
import os
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

folder = r'backend\\data\\amature'
output_csv = r'backend\\data'

# This function reads a file, and extracts the values from it
def readUID(uid):
    df = pd.read_csv(uid, comment='\\', delim_whitespace=True) # Reading the file
    df = df.drop([0,1])
    return df["RELATIVE_FLUX_WITHOUT_SYSTEMATICS|"].tolist()

# This function goes over each file and creates a row for it
def readData(files):
    uidDict = {}
    # Going through each file
    for file in files:
        try:
            base = os.path.basename(file)
            uid = base.split('_')[1]  # '0007562'
            listOfValues = readUID(file)

            # adding value to dictionary
            if not uidDict.get(uid): uidDict[uid] = listOfValues
            else: uidDict[uid].extend(listOfValues)

            print(uid)
        except Exception as e:
            print(f'uid{uid}: Failed due to {e}')
    return uidDict

# This function returns the length, mean, and sd of the exoTrain data
def getExoStats(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['LABEL'])

    # Getting means and std deviations
    row_means = df.mean(axis=1).tolist()
    row_stds  = df.std(axis=1).tolist()

    return 3197, np.mean(row_means), np.mean(row_stds)

def makeCSV(uidDict):

    # Finding max length value in dict
    longest = max(map(len, uidDict.values()))
    print(f'longest length observarion = {longest}')

    # Finding stats about exoTrain data
    exoLen, meanTarget, sdTarget = getExoStats(r"backend\\data\\exoTrain.csv")
    print(f'exoLen = {exoLen}, mean = {meanTarget}, sd = {sdTarget}')
    
    # Re-scaling to be the same as exoTrain shape
    for key in uidDict:
        print(key)
        uidDict[key] = resample(uidDict[key], exoLen)  # Making all lists 3017 in length
        arr = np.array(uidDict[key])
        currentMean = arr.mean()
        currentStd = arr.std()
        transformed = ((arr - currentMean) / currentStd) * sdTarget + meanTarget   # making all means and stds the same as the exoTrain data
        uidDict[key] = transformed.tolist()

    # Finally, making csv
    df = pd.DataFrame.from_dict(uidDict, orient='index')
    df.columns = [f"FLUX.{i}" for i in range(1, df.shape[1] + 1)]
    df.to_csv(r"backend\\data\\amature.csv", index_label="LABEL")

def main():
    files = glob.glob(os.path.join(folder, "*.tbl"))
    uidDict = readData(files)
    print('uidDict created!')
    makeCSV(uidDict)
    print('Data sucessfully exported!')

if __name__ == '__main__':
    main()