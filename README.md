# StellarNet - Hunt for Exoplanets with AI!

StellarNet is project for the NASA Space Apps 2025 Hackathon that allows amateur astronomers to hunt for Exoplanets! The most common way to look for Exoplanets is by collecting light flux data, however, until now it has been difficult to process and interpret this data. StellarNet provides an easy and accurate way to determine if light flux data contains an Exoplanet. Using a custom CNN model, built with PyTorch and trained on both amateur and professional data, users can explore current exoplanet data and upload their own to test!
<img width="1919" height="979" alt="image" src="https://github.com/user-attachments/assets/36116043-3e2f-483e-91c4-0dfe11a10ef4" />


---

## 🚀 Features

- **Interactive Front-End**: Accessible through `index.html` in your browser.
- **Backend API**: Python API (Flask) for data processing and model inference.
- **Machine Learning**: Powered by PyTorch for fast and efficient computations.
- **Flask Intergration**: `flask-cors` enabled for seamless front-end and API communication.
- **Dockerized**: Run everything with a single Docker command—no Python setup required.

---

## 🚀 Model Information  

- **Architecture**: A 1D Convolutional Neural Network specifically trained to analyse time series light flux data.
- **Structure**:

```text
Convolutional Layers:
• Conv1D (1 → 32 channels, kernel=5) + BatchNorm
• Conv1D (32 → 64 channels, kernel=5) + BatchNorm
• Conv1D (64 → 128 channels, kernel=3) + BatchNorm
• MaxPooling (stride=2)
• Dropout (0.3)

Fully Connected Layers:
• FC1 (flattened → 128 units)
• FC2 (128 → 2 classes)
```
- **Training Data**: Trained on two datasources: Lightkurve API Data, collected from Kepler missions and Amateur observations.
- **Data Processing**: All input data is scaled and standardised to ensure consistent predictions! This includes automatically scaling and transforming data uploaded by users for real time inference.
- **Class Imbalance**: The most difficult part about this model was handling class imbalances. There were only a few positive exoplanets and many negative ones. To combat this, I used two techniques:
- **Class Weighting**: Increasing the importance of positive exoplanet examples during training.
- **Data Augmentation**: Generating additional synthetic exoplanet signals to balance the dataset.

---

## 🛠️ Project Structure

StellarNet/  
├─ frontend/ # Web front-end (index.html, CSS, JS)  
├─ backend/ # Python API (api.py) & PyTorch Model (model.py)  
├─ dataPreprocessing/ # Dowloading and processing raw data  
├─ requirements.txt # Python dependencies  
├─ Dockerfile # Docker configuration  
└─ start.sh # Startup script for both servers  

---

## 🛠️ How to Use it


Please download the docker container and run it to test the website! Unfortunately it is not hosted anywhere. 
Alternatively, you can clone the respository yourself and run the following commands to build it yourself:
`git clone https://github.com/<your-username>/StellarNet.git
cd StellarNet
`  
`docker build -t stellarnet .
`  
`docker run -p 8000:8000 -p 5000:5000 stellarnet
`  
`
http://localhost:8000/index.html
`  
---
<img width="1919" height="979" alt="image" src="https://github.com/user-attachments/assets/6536a620-3c27-40cb-a89a-4cc256947e89" />

## 🛠️ User Experience

StellarNet includes a unique, modern, space-themed interface! It contains several features for users including:
  1. Option to explore current Exoplanets: This allows users to scroll through current planets, their light curves, and the models output on them.
     <img width="2866" height="1482" alt="image" src="https://github.com/user-attachments/assets/d9aae6e7-264d-4e39-9091-5e1f5bf0a046" />

  3. Searching for new Exoplanets: This allows users to upload a CSV of any sized light flux and the model will analyse it in real time!
     <img width="2877" height="1468" alt="image" src="https://github.com/user-attachments/assets/586bd125-bdac-4f2b-9a06-d12e3238e15b" />

  5. Feature to learn more about Exoplanets: This links to NASA's page on exoplanets, providing education for users.
  7. Model Analysis Mode: This allows the user to take a look inside the model and see what's going on. 

---
## Thanks for looking at my project!
---

