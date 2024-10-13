# Seasonal_Data_Classification_DL

Notice!!!: Intended for Non-Commercial Use  

## Overview

A binary classification of seasonal data using Deep Learning (CNN+LSTM+Attention).
The model architecture is designed to capture both local features using CNN layers and temporal dependencies using LSTM, while an attention mechanism is employed to focus on relevant parts of the sequence.


```plaintext
├── requirements.txt       # Environment details
├── config/              # Configuration files for model and training parameters
├── model.py               # Defines the CNN+LSTM+Attention model architecture
├── train.py               # Script to train the model and save it to the output folder
├── evaluate.py            # Evaluate a trained model
├── output/                # where trained models are saved
├── results/               # where evaluation results are saved
└── analysis_process.ipynb       # original analysis process, including everything
```

## Requirements

To run this project, you need to install the required dependencies. Use the following command to install them:


```sh
   pip install -r requirements.txt
   ```


## Usage Instructions

The dataset is too large to be stored directly in this repository. You can download the dataset from the following Google Drive link: [Google Drive link](https://drive.google.com/file/d/1tCCQx9c1BHlGrhriokhgtXoMlV7lEZxM/view?usp=sharing).

Place your input dataset in the `data/` folder. The path to this dataset should be specified in `my_config.py` as `dataset_path`.

### Dataset


### Training the Model

To train the model, run the `train.py` script:

```sh
   python train.py
```
### Evaluating the Model

To evaluate a trained model, run the `evaluate.py` script:

```sh
   python evaluate.py
```

## Results

All trained models are saved in the `output/` folder, and evaluation results are saved in the `results/` folder, named with a timestamp

