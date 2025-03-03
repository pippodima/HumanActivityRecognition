# Human Activity Recognition using LSTM and FL

## Overview
This repository implements a Human Activity Recognition (HAR) model using a Long Short-Term Memory (LSTM) network. The model classifies human activities such as walking, sitting, standing, and running based on sensor data from smartphones and smartwatches

## Install dependencies and project
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/pippodima/HumanActivityRecognition.git
cd HumanActivityRecognition
pip install -r requirements.txt
```

## Dataset
The model is trained on publicly available datasets containing labeled sensor data collected from wearable devices.
Ensure you download and preprocess the dataset before training the model:

Dataset can be downloaded [here]((https://dataverse.unimi.it/dataset.xhtml?persistentId=doi:10.13130/RD_UNIMI/QECFKA)) 

After completion of download create two folders `data/raw` and save all the folders `user-1`,
`user-2`, `user-n`, in the raw folder.

then run 
```bash
python preprocess_data/preprocessDataset.py
```

## Configuration  

You can customize the application's settings in `pyproject.toml`:  

```toml
[tool.flwr.app.config]
num-server-rounds = 20  # Total number of server-client training rounds
fraction-fit = 0.5  # Fraction of clients selected for training in each round
local-epochs = 4  # Number of local training epochs per client per round

[tool.flwr.federations.local-simulation]
options.num-supernodes = 25  # Total number of clients
```
## Run with the Simulation Engine

In the `project` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```