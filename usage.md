# How does Transformer model evolve to learn diverse chemical structres?
Investigation on procedure of Transformer learning chemical structure.   
This document describes the codes used in the experiments and how to replicate them.

## Environments
The following packages are required. Note that the latest versions have now possibly become incompatible with our codes, and older versions within these ranges may work.
- python>=3.8
- numpy>=1.19.2
- pandas>=1.2.5
- tqdm>=4.51.0
- PyTorch  
    Version 1.8 was used for most of the experiments, but 1.10 was used for the model with pre-LN structure. Either version works for our codes unless pre-LN is adopted.

You can build a conda environment using [requirements.txt](requirements.txt) except PyTorch like following commands. For PyTorch package, please install the proper version according to your GPU environment.
```sh
conda create -n chirality python==3.8
conda activate chirality
conda install pip
pip install -r requirements.txt
```
Setting environment variable ```PROJECT_DIR``` is recommended.
```sh
export PROJECT_DIR=/path/to/ChrailityMisunderstanding
```


## Data preprocessing for training
First of all, tokenization of SMILES is required for training.
```sh
python preprocess/tokenize_.py --smiles data/example_data.csv --output data/example
```

## Training
The Transformer model can be trained with tokenized data.
```sh
python experiments/training.py --studyname example --train_file data/example --val_file data/example.pkl 
```

## Trained model weights
Model weights trained in several conditions are prepared.
- Fully trained model (step 80,000) in training where stagnation did not occur (weights/no_stagnation.pth).
- Fully trained model (step 80,000) in training where stagnation occurred (weights/stagnation.pth).
- Fully trained model (step 80,000) with pre-LN structure (weights/preln.pth).

## Evaluation
Perfect / partial accuracy of translation can be calculated for any SMILES data you have. Predicted SMILES can also be stored.
```sh
python experiments/evaluate.py --model_file weights/no_stagnation.pth --smiles data/example_data.csv --input_col random --target_col canonical --output data/prediction_example.csv
```
Specify ```--preLN``` option when you use a weight with pre-LN structure.

## Featurization
You can featurize molecules with the trained model. Tokenization is not required.
```sh
python experiments/featurize.py --model_file weights/no_stagnation.pth --smiles data/example_data.csv --col canonical --output data/example_feature.csv
```
Specify ```--preLN``` option when you use a weight with pre-LN structure.