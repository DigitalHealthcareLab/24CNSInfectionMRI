# 24CNSInfectionMRI

## Overview

This README contains all information on how to run the pipeline for *Multimodal Deep Learning with MRI-Clinical Integration for Prediction of Prognosis in Central Nervous System Infections*.

Below is the graphical abstract of the study:
![](/_images/Graphical_Abstract.png)

## Introduction

CNS Infections are inflammatory conditions involving the membranes and parenchyma of the CNS including meningitis and encephalitis. Early treatment is often empirical, as mistreatment may lead to possibly fatal side effects such as acute kidney injury and liver damage, thus highlighting the need for early prognosis prediction. Efforts have been made to predict the prognosis in encephalitis; However, most of them have focused on predicting the prognosis associated with a single pathogen and are limited to specific situations. We aim to develop a deep learning model for the early prognosis prediction of CNS inflammation with multimodal data including clinical features and brain imaging data. 

## Installation

To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```
## Usage

The pipeline is divided into 5 main steps. Each step should be run concecutively, and the instructions for each step is detailed in their respective folders.

Please refer to the README in each folder for more information.

Acknowledgements to the authors of the code used in this pipeline are provided in the README of each folder.

__The description of each step is as follows:__

### 1. MRI Data Preprocessing
![](/_images/Step_1.png)
Within directory `1_MRI_Preprocessing`: the MRI data is segmented and preprocessed. The MRI data is then saved in the appropriate format for the next step.

### 2. Unimodal MRI Model Training
![](/_images/Step_2.png)
Within directory `2_Unimodal_MRI_CNN`: the MRI data for each brainpart is used to train unimodal models for each part. The models are then utilized to extract vectorized features from the MRI data. The extracted features are saved in the appropriate format for the next step.

### 3. Data Vectorization
![](/_images/Step_3.png)
Within directory `3_Data_Vectorization`: the clinical data is vectorized and saved in the appropriate format for the next step.

### 4. Multimodal Fusion Model Training
![](/_images/Step_4.png)
Within directory `4_Multimodal_Fusion`: the vectorized clinical and MRI data are used to train a multimodal fusion model. The model is then evaluated and saved in the appropriate format for the next step.

### 5. Model Evaluation & Visualization
![](/_images/Step_5.png)
Within directory `5_Model_Evaluation`: the model is evaluated and visualized.

## Citation
If you find this work useful, please cite the following paper:
(To be updated)