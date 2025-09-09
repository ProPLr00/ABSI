# ABSI
 
 By ProPLr00 (Jun Wen NG), and other dailous who carry me along the way.

 Nanyang Technological University

### Table of Contents
0. [Introduction](#introduction)
1. [Citation](#citation)
2. [Enviromental Setup](#environment-setup)
3. [Data](#data)
4. [Code](#code)

### Introduction
This framework - `Adaptive Bayesian Machine Learning with SISSO Interpretation (ABSI)` is dedicated for the explainable-adaptive machine learning pipeline in the paper "Accelerated Discovery of `CrTe Nanowire` with Adaptive Machine Learning" (arxiv link). These models are used for iterative improvement of the `CrTe Nanowire` length task.

### Citation
If you are going to use these model in your research, please cite: (arxiv link)

### Environment Setup
Please check the env folder for respective env required for the conda enviroment.
0. For the `Adaptive_BO.ipynb` and other python files, the env required will be the `enviroment_MT.yml`.
1. For the `SISSO_Analysis.ipynb`. the env required will be the `environment_sisso.yml`.

### Data

0. For exact model weight and seed to make sure the per iteration result is repeatable, please refer to [model weight](OneDriveLink).

1. For more detailed description of the dataset, please check out our [paper](#introduction).

### Code
0. Code Structure
- **results** :`folder to store all results and generated figures`
- **data** : `download data before running code `(see [Data](#code))
- **env** : `for the env setup for different part of the code`(see [Environment Setup](#environment-setup))
- Adaptive_BO.ipynb : `for the adaptive Bayesian Optimization part.`
- SISSO_Analysis.ipynb : `for the SISSO analysis part.`
- common_splits.py:  `for the initial seed generation (common ground), then the pkl file will load for the following`
- Simplified_test.py - `for choosing and hyperparameter tuning of the Bayesian Acquisition Function.`
- Simplified_test.py - `for the traditional GP-LCB test.`


**Extra Notes:**
0.  For random seed generation, directly run the simplified_test_*.py will do (check the commented section in the .py files).