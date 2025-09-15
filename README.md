# ABSI
 
 By ProPLr00 (Jun Wen NG), and other dailous who carry me along the way.

 Nanyang Technological University

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Enviromental Setup](#environment-setup)
0. [Data](#data)
0. [Code](#code)

### Introduction
This framework - `Adaptive Bayesian Machine Learning with SISSO Interpretation (ABSI)` is dedicated for the explainable-adaptive machine learning pipeline in the paper "Accelerated Discovery of `CrTe Nanowire` with Adaptive Machine Learning" (arxiv link). These models are used for iterative improvement of the `CrTe Nanowire` length task.

### Citation
If you are going to use these model in your research, please cite: (arxiv link)

### Environment Setup
Please check the env folder for respective env required for the conda enviroment.

0. For the `Adaptive_BO.ipynb` and other python (.py) files, the env required will be the `enviroment_MT.yml`.

0. For the `SISSO_Analysis.ipynb`. the env required will be the `environment_sisso.yml`.

### Data

0. For exact model weight and seed to make sure the per iteration result is repeatable, please refer to [model weight](OneDriveLink).

0. For more detailed description of the dataset, please check out our [paper](#introduction).

### Code
0. Code Structure
- **results** :`folder to store all results and generated figures`
- **data** : `download data before running code `(see [Data](#data)), `if you are using another dataset please put inside and change the **name** in the respective python files`
- **env** : `for the env setup for different part of the code`(see [Environment Setup](#environment-setup))

0. For the ABSI framework:

- Adaptive_BO.ipynb : `Adaptive Bayesian Machine Learning, AB part of this project, consist of 5 parts`
- - Best Model Hyperparameter Generation - `Visualization of data and generation of best hyperparameter through nested-cross validation` - `results/best_model/best_model_*.pkl`
- - Shuffled Model Generation - `Generation of shuffled model based on best model_*.pkl and shuffled dataset` - `results/shuffled_model/shuffled_model_*.pkl`
- - Materials Search Space Generation - `Definition of the materials search space` - `results/MSS_name_*.csv`
- - Materials Search Space Prediction - `Prediction of the shuffled models on the materials search space` - `results/chunk_metrics/chunk_metrics_*.csv`
- - Top Acquisition Sampling - `Extraction of the top sampling score candidates from the overall chunk_metrics_*.csv` - `results/top100_acq_func_*.csv`
- - Mixture of Acquisition Sampling (future extension) - `Extraction of top candidates which recommendation by 2 or more acquisition function` - `results/repeated_candidates_*.csv`

- SISSO_Analysis.ipynb : `SISSO Interpretation based on` [TorchSISSO](https://github.com/PaulsonLab/TorchSISSO) library`, SI part of this project, consist of 3 parts`
- - Settings - `Basic settings of the SISSO hyperparameter, patches for the TorchSISSO library, and helper functions`
- - Leave-One-Out Cross Validation - `Generation of the best hyperparameter through LOOCV and preliminary descriptor frequency analysis` - `results/sisso_loocv_*.csv`
- - Out-of-Bag Cross Validation - `Selection of best descriptor based from extra-OOBCV based on the estimation of how well the descriptor generalized` - `results/sisso_bootstrap*.csv`

0. Selection of the best acquisition function:

- common_splits.py:  `Visualization and generation of the common seed splits for comparison of the acquisition functions; adjust num_splits for the number of common splits; adjust the init_train_size for the size of each splits` - to run: `python common_splits.py`- `data/common_split_*.pkl`

- Simplified_test.py - `Performance prediction/testing for the candidate acquisition functions (UF); adjust outer_loop for the number of repetition` - to run: `python Simplified_test.py --UF "acquisition function"` - `results/folder_with_result/*`

- Simplified_test_GP.py - `Performance prediction/testing for the traditional GP-LCB test; ; adjust outer_loop for the number of repetition` - to run: `python Simplified_test_GP.py` - `results/GP_LCB_name/*`

- Analysis_Simplified_test.ipynb - `Visualization of the performance for the candidate acquisition function`


**Extra Notes:**

- ModelSelection_Regression.py - `Adjusted based on the` [ML-guided-material-synthesis](https://github.com/MSwML/ML-guided-material-synthesis) `for the regression model selection, adjust the settings inside for another usage`.

- For random seed generation, directly run the simplified_test_*.py will do (check the commented section in the .py files).
