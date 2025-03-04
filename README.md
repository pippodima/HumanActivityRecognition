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

## Model Saving Configuration  

In `server.py`, there is a boolean variable at the beginning of the script that controls whether the model should be saved. Set this variable to `True` if you want to save the model, or `False` if you don’t.  

Additionally, specify the number of rounds after which the model should be saved.  

Example:  
```python
save_model = True  # Set to True to enable model saving, False to disable

save_model_weights(parameters, round_num, save_every=10) # Define after how many rounds the model should be saved
```

## Plotting Metrics  

At the end of each training session, a final summary is printed in the console. This summary includes information on training rounds, loss history, and evaluation metrics such as accuracy, F1-score, precision, and recall.  

### Example Output:  


```
INFO :      [SUMMARY]
INFO :      Run finished 5 round(s) in 177.59s
INFO :          History (loss, distributed):
INFO :                  round 1: 2.349399052941855
INFO :                  round 2: 2.3358307605865183
INFO :                  round 3: 1.8299632479962824
INFO :                  round 4: 1.7774486594872922
INFO :                  round 5: 1.8502811750539483
INFO :          History (metrics, distributed, fit):
INFO :          {'avg_train_loss': [(1, 1.4637292970239726),
INFO :                              (2, 0.9909953764905597),
INFO :                              (3, 1.3720482810227959),
INFO :                              (4, 0.8500210718050891),
INFO :                              (5, 0.9267523147506651)]}
INFO :          History (metrics, distributed, evaluate):
INFO :          {'avg_accuracy': [(1, 0.30856601465492256),
INFO :                            (2, 0.3520497184316254),
INFO :                            (3, 0.38676684574886994),
INFO :                            (4, 0.3700034737435145),
INFO :                            (5, 0.45672350411248813)],
INFO :           'avg_f1_score': [(1, 0.2483657930050348),
INFO :                            (2, 0.3357100433216376),
INFO :                            (3, 0.39009850309539756),
INFO :                            (4, 0.36271641005919825),
INFO :                            (5, 0.4696033190615891)],
INFO :           'avg_precision': [(1, 0.3206848098272536),
INFO :                             (2, 0.4027330013216786),
INFO :                             (3, 0.5559130964917155),
INFO :                             (4, 0.4938532002879052),
INFO :                             (5, 0.5673340645779819)],
INFO :           'avg_recall': [(1, 0.30856601465492256),
INFO :                          (2, 0.3520497184316254),
INFO :                          (3, 0.38676684574886994),
INFO :                          (4, 0.3700034737435145),
INFO :                          (5, 0.45672350411248813)]}
INFO :      
```


### How to Plot Metrics  
1. Copy the summary from the console.  
2. Create a `.txt` file and paste the summary into it.  
3. Save the file inside `plot/rawLogs/`.  
4. Run the following command to generate the plots:  

```sh
python plotting.py -f filepath
```
Replace `filepath` with the actual path to your `.txt` file.
To save the generated plots, add the `-s` flag:

``` sh
python plotting.py -f filepath -s
```

This will generate visual representations of the training metrics, making it easier to analyze performance trends