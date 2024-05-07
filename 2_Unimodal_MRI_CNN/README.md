# Overview
![](/_images/Step_2.png)
This folder contains the code for training and evaluating a CNN model for each brainpart using 4-fold cross-validation.

# Requirements

# Usage

## Step 0. Construct internal and external datasets
- Construct internal and external datasets from the outputs of step 1.
- Specifically, retrieve the indices of the internal and external datasets from the outputs of step 1.
- Save the datasets to `./data/`.
Run files in folder `/0_Labels_Recon`.

## Step 1. Construct Fold Indices
- Construct fold indices for 4-fold cross-validation.
- Save fold indices to `./data/fold_indices.pkl`.
Run notebook `1_Construct_Fold_Indices.ipynb`.

## Step 2. Train CNN
- Train a CNN model for each brainpart using 4-fold cross-validation.
- Save trained model to `./models/`.
Save your configurations to `/home/yhchoi/24CNSInfectionMRI/2_Unimodal_MRI_CNN/src/_cnn_config.py`, then run python file `2_CNNTrain_KFold.py`. Repeat with different configurations if model underperforms.

## Step 3. Evaluate CNN and Save fold with best performance along with test results
- Evaluate the trained CNN model based on training results.
- Save fold with best performance along with test results to csv, for each brainpart.
Run notebook `3_Readlogs_Testset.ipynb`. Note that the configurations should be the same as the configurations used in training.

## Step 4. Extract the best models
- Extract the best models for each part and save it to `./models/`.
Run notebook `4_Best_Models_Extraction.ipynb`.

## Step 5. Extract features using the best models on the internal dataset
- Extract features using the best models on the internal dataset.
- Save the features to `./features/`.
Run notebook `5_CNN_Inference_Vectoroutput_Internal.py`.

## Step 6. Extract features using the best models on the external dataset
- Extract features using the best models on the external dataset.
- Save the features to `./features/`.
Run notebook `6_CNN_Inference_Vectoroutput_External.py`.