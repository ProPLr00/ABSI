import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import os
import pickle

import datetime
import time
import random

from scipy.stats import pearsonr, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, ndcg_score
from sklearn.model_selection import KFold, GridSearchCV
from skopt.acquisition import gaussian_ei, gaussian_lcb

import utils

#handle warnings
import warnings
warnings.filterwarnings("ignore")
np.random.seed(3)
np.set_printoptions(precision=5)

name = "CrTeNW_data"

root_dir = str(Path(os.getcwd()))
from_dir = root_dir + "/data/"
to_dir = root_dir + "/results/"

# Settings
outer_loop = 10
inner_loop = 1

# BO cq_func: "EI". "LCB", "MOCU_mean", "MOCU_std_dev", "rank_score", "hybrid"
UF = "LCB" 

tuned_parameters = dict(learning_rate=[0.01],
                    n_estimators=[300,500,700],
                    colsample_bylevel=[0.5,0.7,0.9],
                    gamma=[0,0.2],
                    max_depth=[3,7,11],
                    reg_lambda=[0.1,1,10],
                    subsample=[0.4,0.7,1])

# Main BO Functions
def PAM_regression(acq_func="EI", save_csv=False, verbose=False, to_break=True, title="default_title", batch=1,
                   exploration_ratio = 0.1, kappa_base=1.96, gamma = 0.2, alpha = 0.9, beta = 0.1, m_shuffles=3, 
                   n_jobs=4, init_train_ind=None, init_test_ind=None):
    """
    Unified PAM-guided synthesis function that supports multiple acquisition functions.

    Parameters:
        acq_func (str): Acquisition function to use. Options: "EI", "LCB", "MOCU_mean", "MOCU_std_dev", "rank_score".
        save_csv (bool): Whether to save the results as a CSV file.
        verbose (bool): Whether to print detailed iteration info.
        to_break (bool): Stop once the global maximum is reached (if found).
        title (str): Title for saving results (e.g., filename prefix).
        batch (int): Number of candidates to add per iteration.
        exploration_ratio (float): Multiplier for EI’s `xi_relative`.
        kappa_base (float): Base exploration parameter for LCB.
        gamma (float): Tuning factor for how much uncertainty ranking should influence LCB’s kappa.
        alpha (float): Weight for prediction ranking in "rank_score".
        beta (float): Weight for uncertainty ranking in "rank_score".
        m_shuffles (int): Number of ensemble members (bootstrapped models).
        n_jobs (int): Number of jobs for parallel processing (e.g., in GridSearchCV).
        init_train_ind (list, optional): Predefined initial indices for the training set.
        init_test_ind (list, optional): Predefined initial indices for the testing set.

    Returns:
        list: A list of the following elements:
              1. saved_title (str): Path or identifier for the saved CSV (or '-' if not saved).
              2. Nc (int): Iteration at which the global max was found (0 if never found).
              3. mean_y_wo_init (float): Placeholder (currently np.nan).
              4. std_y_wo_init (float): Placeholder (currently np.nan).
              5. mean_y_w_init (float): Mean of the final training set yields.
              6. std_y_w_init (float): Standard deviation of the final training set yields.
              7. mean_y_only_init (float): Mean of the initial training set yields.
              8. std_y_only_init (float): Std of the initial training set yields.
              9. run_time (float): Total run time in minutes.
              10. acq_history (list): List of acquisition-value arrays (one per iteration).

    Notes:
        - This function references global variables such as `X`, `Y`, `all_ind`, etc.
          They must be defined in the global scope or replaced with parameters.
        - The XGBoost version is assumed to be 0.8.0, so we keep `objective="reg:linear"`.
    """
    init_time = time.time()
    Nc = 0

    # From common_splits.py or generate randomly here
    if init_train_ind is not None and init_test_ind is not None:
        train_ind = init_train_ind.copy()
        test_ind = init_test_ind.copy()
    else:
        train_ind = random.sample(all_ind_wo_max, init_train_size)
        test_ind = [x for x in all_ind if x not in train_ind]

    # Allocate results matrix for 10 columns, depends on inner split size
    results_mat = np.empty((totalSamp - init_train_size, 10), dtype=object)
    if verbose:
        print('Initial training set indexes:', train_ind)

    j = 0
    loop_count = 0
    mean_y_only_init = np.mean(Y[train_ind])
    std_y_only_init = np.std(Y[train_ind])
    
    # List to store full acquisition vectors from each iteration
    acq_history = []
    
    while j < (totalSamp - init_train_size):
        X_train = X[train_ind]
        Y_train = Y[train_ind]
        X_test = X[test_ind]
        Y_test = Y[test_ind]
        
        train_size = len(Y_train)
        last_max = np.max(Y_train)
        
        # Step 1: Hyperparameter tuning using GridSearchCV.
        inner_cv = KFold(n_splits=inner_nsplits, shuffle=True, random_state=j)
        reg = xgb.XGBRegressor(objective="reg:linear", min_child_weight=1, **{'tree_method': 'exact'},
                                silent=True, n_jobs=n_jobs, random_state=3, seed=3)
        gb_clf = GridSearchCV(reg, tuned_parameters, cv=inner_cv, scoring='r2', verbose=0, n_jobs=n_jobs)
        gb_clf.fit(X_train, Y_train)
        best_params = gb_clf.best_estimator_.get_params()
        
        # Step 2: Build ensemble on bootstrapped samples.
        ensemble_preds = []
        for s in range(m_shuffles):
            bootstrap_idx = np.random.choice(range(len(X_train)), size=len(X_train), replace=True)
            X_boot = X_train[bootstrap_idx]
            Y_boot = Y_train[bootstrap_idx]
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_boot, Y_boot)
            ensemble_preds.append(model.predict(X_test))
        ensemble_preds = np.array(ensemble_preds)
        ensemble_mean = np.mean(ensemble_preds, axis=0)
        ensemble_std = np.std(ensemble_preds, axis=0)
        # Step 3: Best observed yield.
        f_best = np.max(Y_train)

        # Step 4: Compute acquisition values based on the chosen function.
        # Define a dummy model to work with skopt's functions.
        class DummyModel:
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std
            def predict(self, X, return_std=False):
                if return_std:
                    return self.mean, self.std
                return self.mean
        
        dummy_model = DummyModel(-ensemble_mean, ensemble_std)  # For EI and LCB we use negatives.
        candidates = np.expand_dims(-ensemble_mean, axis=1)
        acq_vals = None
        
        if acq_func == "EI":
            y_opt = -f_best
            avg_ensemble_std = np.mean(ensemble_std)
            xi_relative = avg_ensemble_std * exploration_ratio
            acq_vals = gaussian_ei(candidates, dummy_model, y_opt=y_opt, xi=xi_relative, return_grad=False)

        elif acq_func == "LCB":
            # For LCB, we use gaussian_lcb: lcb = -ensemble_mean - kappa * ensemble_std.
            acq_vals = gaussian_lcb(candidates, dummy_model, kappa=kappa_base, return_grad=False)
            # For the hybrid function between LCB and rank use this
            '''rank_uncertainty = rankdata(ensemble_std, method='max')
            N = len(ensemble_std)
            kappa_candidates = kappa_base * (1 + gamma * (rank_uncertainty / N))'''
    
        elif acq_func == "MOCU_mean":
            # Greedy selection based solely on the ensemble mean (highest mean).
            acq_vals = ensemble_mean  

        elif acq_func == "MOCU_std_dev":
            # Select candidate with highest uncertainty (std)
            acq_vals = ensemble_std 

        elif acq_func == "rank_score": 
            # Rank_score method from the HEA
            rank_pred = rankdata(ensemble_mean, method='max')
            rank_uncertainty = rankdata(ensemble_std, method='max')
            alpha = alpha
            beta = beta
            acq_vals = alpha * rank_pred + beta * rank_uncertainty
                        
        else:
            raise ValueError("Unknown acquisition function: {}".format(acq_func))
        
        # Save the full acquisition vector for this iteration.
        acq_history.append(acq_vals)
        
        # LCB, lower values are better if we are minimizing; 
        # For MOCU_mean, higher is better, and for MOCU_std_dev, higher is better.
        if acq_func in ["LCB"]:
            sorted_inds = np.argsort(acq_vals)  # ascending order
        else:
            sorted_inds = np.argsort(-acq_vals)  # descending order
        
        best_pos_inds = sorted_inds[:batch] 
        best_pred = ensemble_mean[best_pos_inds]

        # Step 5: Evaluate ensemble predictions.
        r2 = r2_score(Y_test, ensemble_mean)
        mse = mean_squared_error(Y_test, ensemble_mean)
        pear, p_value = pearsonr(Y_test, ensemble_mean)
        try:
            ndcg = ndcg_score([Y_test], [ensemble_mean], k=batch)
        except Exception:
            ndcg = np.nan
        
        next_best_true_ind = [test_ind[i] for i in best_pos_inds]
        next_best_y_true = Y_test[best_pos_inds]
        best_acq_value = acq_vals[best_pos_inds]
            
        result_list = [train_size,
                        str(next_best_true_ind),
                        str(best_pred),
                        str(next_best_y_true),
                        r2, mse, pear, p_value, ndcg,
                        str(best_acq_value)
                        ]
        results_mat[loop_count, :] = np.array(result_list)
        
        loop_count += 1
        j += batch
        
        if verbose:
            print(loop_count, '->', j, f"Train Size = {len(train_ind)}",', best_next_ind=', next_best_true_ind, 
                ' best_Y_true=', np.round(next_best_y_true, 6),
                ' train_max=', "{0:.6f}".format(last_max), 
                ' r2=', r2, f'{acq_func}_val: {best_acq_value.tolist()}'
                #,"Ensemble Mean:", ensemble_mean, "Ensemble Std:", ensemble_std
                )

        # Update training and test indices.
        train_ind.extend(next_best_true_ind)
        test_ind = [x for x in test_ind if x not in next_best_true_ind]
        
        if (next_best_y_true == Y_global_max).any() and Nc == 0:
            Nc = j + init_train_size
            if to_break:
                break
                
    saved_title = '-'
    if save_csv: 
        results = pd.DataFrame(data=results_mat[0:loop_count, :],
                            columns=['sample_size', 'pred_ind', 'best_pred_result', 'y_true',
                                        'r2', 'mse', 'pearson', 'p_value', 'ndcg', 'acq_value'])
        saved_title = utils.save_csv(results, title=title)
        
    mean_y_wo_init = np.nan
    std_y_wo_init = np.nan
    mean_y_w_init = np.mean(Y[train_ind])
    std_y_w_init = np.std(Y[train_ind])
    run_time = (time.time() - init_time) / 60
    
    return [saved_title, Nc, mean_y_wo_init, std_y_wo_init, mean_y_w_init, std_y_w_init,
            mean_y_only_init, std_y_only_init, run_time, acq_history]

df_cleaned, X, Y, clean_feature_list, clean_result_col = utils.load_and_clean_data(name, target="length")

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X = X_normalized

inner_nsplits = 10
init_train_size = 20
totalSamp = X.shape[0]

# Identify the index of the global maximum from Y
global_max_ind = np.argmax(Y)
Y_global_max = Y[global_max_ind]

# Create a full list of indices and exclude the global max for training
all_ind = list(range(totalSamp))
all_ind_wo_max = [i for i in all_ind if i != global_max_ind]

# Load the common splits from file (assume they were generated earlier) for comparison
splits_filepath = os.path.join(from_dir, "common_splits_1.pkl")
with open(splits_filepath, 'rb') as f:
    common_splits = pickle.load(f)

### For random generation of split, please use here ###
'''# Instead of loading common splits, generate them randomly:
common_splits = []
num_splits = 10  # number of random splits

for _ in range(num_splits):
    # Sample training indices from all_ind_wo_max so that the global max is never in the training set
    train_ind = random.sample(all_ind_wo_max, init_train_size)
    # Test indices are generated from all_ind, so the global max (global_max_ind) is included
    test_ind = [x for x in all_ind if x not in train_ind]
    common_splits.append((train_ind, test_ind))'''

# Print the generated splits for verification
for split in common_splits:
    print(split)

# Save the splits to a file.
today = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
splits_filepath = os.path.join(from_dir, f"common_splits_random_{today}.pkl")
with open(splits_filepath, 'wb') as f:
    pickle.dump(common_splits, f)
print("Initial splits saved to:", splits_filepath)

print('start PAM for ', str(outer_loop * inner_loop * len(common_splits)), ' times...')

all_acq_history = []  # Aggregate acquisition history from each run
res_arr = []          # Aggregate summary results (without full acquisition history)
all_results = []      # Aggregate full results from each run

# Outer and inner loops
for j in range(outer_loop):
    init_time_outer = time.time()
    for i in range(inner_loop): # As inner loop always = 1 so can ignore its effect here, but maintain for future application/usage
        loop_count = j * inner_loop + i        
        # Iterate over each common split
        for split in common_splits:
            train_ind, test_ind = split        
            result = PAM_regression(
                acq_func= UF,  
                save_csv=True, 
                verbose=True, 
                to_break=True, 
                title=name + f"_{UF}_random_run", 
                batch=1, 
                exploration_ratio=0.1, #for EI
                kappa_base=1.44, # for LCB
                #gamma = (1.44/1.282) - 1,   # for hybrid LCB_rank
                alpha = 0.9, beta = 0.1, #for rank score
                m_shuffles=10, #increase for more initial dataset shuffling
                n_jobs=6,
                init_train_ind=train_ind, 
                init_test_ind=test_ind
            )
            all_results.append(result)
            res_arr.append(result[:-1])
            all_acq_history.append(result[-1])
            print(str(loop_count), ' -> ', str(result[0]), '  time=', result[-2])
   
    PAM_df = pd.DataFrame(data=res_arr, columns=[
        'file-name','num_experiments','mean_y_wo_init','std_y_wo_init',
        'mean_y_w_init','std_y_w_init','mean_y_only_init','std_y_only_init','run_time'
    ])
    saved_path = utils.save_csv(PAM_df, title=name + str(inner_loop) + 'times_')
    
    # Save results
    now_time = datetime.datetime.now()
    today = now_time.strftime("%Y_%m_%d_%H%M%S")
    acq_history_path = os.path.join(to_dir, name + f"{UF}_acq_history_{today}.pkl")
    with open(acq_history_path, 'wb') as f:
        pickle.dump(all_acq_history, f)
    print("Acquisition history saved to:", acq_history_path)
    print('Total time for outer loop iteration:', str((time.time() - init_time_outer) / 3600), ' hrs  >>-------saved')