"""
This is the API which allows users to upload files for inference
- It scales and standardises all files so that they are compatible with the model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import json
import torch.nn.functional as F
import pandas as pd
from scipy.signal import resample

app = Flask(__name__)
CORS(app)

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

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(3197,2).to(device)
model.load_state_dict(torch.load('backend/modelMixed.pth', map_location=device))
model.eval()

# This function returns the length, mean, and sd of the exoTrain data
def getExoStats(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['LABEL'])

    # Getting means and std deviations
    row_means = df.mean(axis=1).tolist()
    row_stds  = df.std(axis=1).tolist()

    return 3197, np.mean(row_means), np.mean(row_stds)

# This function scales the data so that it works with the model no matter what
exoLen, meanTarget, sdTarget = getExoStats(r"backend\\data\\exoTrain.csv")
print(f'exoLen = {exoLen}, mean = {meanTarget}, sd = {sdTarget}')

# Re-scaling to be the same as exoTrain shape
def scaleData(flux_data):
    if len(flux_data) != 3197:
        flux_data = resample(flux_data, exoLen)  # Making all lists 3017 in length
        arr = np.array(flux_data)
        currentMean = arr.mean()
        currentStd = arr.std()
        transformed = ((arr - currentMean) / currentStd) * sdTarget + meanTarget   # making all means and stds the same as the exoTrain data
        flux_data = transformed.tolist()
    return flux_data



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        flux_data = data['fluxData']
        flux_data = scaleData(flux_data)
        
        # Convert to tensor
        flux_tensor = torch.FloatTensor([flux_data]).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(flux_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            certainty = probabilities[0][predicted_class].item()
        
        # Map to class names
        class_names = ['Non-Planet', 'Planet']
        
        return jsonify({
            'success': True,
            'predictedClass': class_names[predicted_class],
            'predCertainty': certainty,
            'fluxData': flux_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  # Note: remove host arg if things break
