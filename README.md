# ABSI
 Adaptive Bayesian Machine Learning with SISSO Interpretation

To run this code:
1. Adaptive_BO.ipynb - for the adaptive Bayesian Optimization part.
2. SISSO_Analysis.ipynb - for the SISSO analysis part.

For the benchmarking:
1. Run the common_splits.py for the initial seed generation (common ground), then the pkl file will load for the following:
    a. Simplified_test.py - for choosing and hyperparameter tuning of the Bayesian Acquisition Function.
    b. Simplified_test.py - for the GP-LCB test.
2. For random seed generation, directly run the simplified_test_*.py will do (check the commented section).