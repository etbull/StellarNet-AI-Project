# StellarNet - Hunt for Exoplanets with AI!

StellarNet is project for the NASA Space Apps 2025 Hackathon that allows amateur astronomers to hunt for Exoplanets! The most common way to look for Exoplanets is by collecting light flux data, however, until now it has been difficult to process and interpret this data. StellarNet provides an easy and accurate way to determine if light flux data contains an Exoplanet. Using a custom CNN model, built with PyTorch and trained on both amateur and professional data, users can explore current exoplanet data and upload their own to test!

---

## 🚀 Features

- **Interactive Front-End**: Accessible through `index.html` in your browser.
- **Backend API**: Python API (Flask) for data processing and model inference.
- **Machine Learning**: Powered by PyTorch for fast and efficient computations.
- **Flask Intergration**: `flask-cors` enabled for seamless front-end and API communication.
- **Dockerized**: Run everything with a single Docker command—no Python setup required.

---

  ## 🚀 Model Information  

- **Architecure**: A 1D Convolutional Neural Network specifically trained to analyse time series light flux data.
- **Structure**
      ```Convolutional Layers:
    • Conv1D (1 → 32 channels, kernel=5) + BatchNorm
    • Conv1D (32 → 64 channels, kernel=5) + BatchNorm
    • Conv1D (64 → 128 channels, kernel=3) + BatchNorm
    • MaxPooling (stride=2)
    • Dropout (0.3)
    
    Fully Connected Layers:
    • FC1 (flattened → 128 units)
    • FC2 (128 → 2 classes)```
- **Training Data**: Trained on two datasources Lightkure API Data, collected from Kepler missions and Amateur observations.
- **Data Processing**: All input data is scaled and standardised to ensure consisten predictions! This includes automatically scalling and transforming data uploaded by users for real time inference.
- **Class Imbalance**: The most difficult part about this model was handling class imbalances. There were only a few positive exoplanets and many negatve ones. To combat this, I used two techniques
    1. Class Weighting: Increasing the importance of positive exoplanet examples during training.
    2. Data Augmentation: Generating additional synthetic exoplanet signals to balance the dataset. 

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
Alternatively, you can clone the respository yourself and run the following command to build it yourself:
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
## Thanks for looking at my project!
---

