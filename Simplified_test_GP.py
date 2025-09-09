import os, time, random, datetime, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from sklearn.metrics import r2_score, mean_squared_error, ndcg_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

import utils    

warnings.filterwarnings("ignore")
np.random.seed(3)
np.set_printoptions(precision=5)

# Main GP function
def PAM_regression_GP(kappa=1.44, *, save_csv=False, verbose=False,
                      m_restarts=10, init_train_ind, init_test_ind,
                      title="gp_run", to_break=True, batch=1):
    start = time.time()
    train_ind, test_ind = init_train_ind.copy(), init_test_ind.copy()
    results, acq_history = [], []

    # Until candidate exhausted
    while test_ind:                               
        X_train, y_train = X[train_ind], Y[train_ind]
        X_test,  y_test  = X[test_ind],  Y[test_ind]

        # GP fit (Matern-5/2, ARD)
        d = X_train.shape[1]
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d),
                   length_scale_bounds=(1e-2, 1e2), nu=2.5) \
                 + WhiteKernel(noise_level=1e-6,
                               noise_level_bounds=(1e-9, 1.0))
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       n_restarts_optimizer=m_restarts,
                                       normalize_y=True,
                                       random_state=3)
        gpr.fit(X_train, y_train)

        mu, sigma = gpr.predict(X_test, return_std=True)

        # LCB acquisition
        acq = mu - kappa * sigma
        acq_history.append(acq)                     # store full vector
        best_local = np.argsort(acq)[:batch]        # minimise LCB
        next_idx   = [test_ind[i] for i in best_local]

        # Diagnostics
        r2   = r2_score(y_test, mu)
        mse  = mean_squared_error(y_test, mu)
        pear, p_val = pearsonr(y_test, mu)

        results.append([len(train_ind), next_idx, mu[best_local],
                        y_test[best_local], r2, mse, pear, p_val,
                        acq[best_local]])

        if verbose:
            best_true = y_test[best_local]
            print(f"{len(train_ind):3d} → add {next_idx}, y* = {best_true}")

        train_ind.extend(next_idx)
        test_ind = [ix for ix in test_ind if ix not in next_idx]

        if to_break and (Y[next_idx] == Y_global_max).any():
            break                                  # early success

    run_min = (time.time() - start) / 60
    saved = "-"
    if save_csv:
        cols = ["train_size", "picked_idx", "mu_pred", "y_true",
                "r2", "mse", "pear", "p_val", "acq_val"]
        saved = utils.save_csv(pd.DataFrame(results, columns=cols),
                                      title=title)

    return [saved, len(train_ind), np.nan, np.nan,
            np.mean(Y[train_ind]), np.std(Y[train_ind]),
            np.mean(Y[init_train_ind]), np.std(Y[init_train_ind]),
            run_min, acq_history]

# Parameters
DATASET_NAME  = "CrTeNW_data"
KAPPA         = 1.44                   
OUTER_LOOP    = 10
INNER_LOOP    = 1           # As the common splits is already being introduced here
INIT_TRAIN_SZ = 20

root_dir = Path.cwd()
from_dir = root_dir / "data"
to_dir   = root_dir / "results" / "GP-BO_LCB" 
to_dir.mkdir(parents=True, exist_ok=True)

# Data Loading and setting
df_cleaned, X, Y, feat_cols, tgt_col = utils.load_and_clean_data(DATASET_NAME,target="length")

scaler = StandardScaler()
X      = scaler.fit_transform(X)

total_samp      = X.shape[0]
global_max_ind  = int(np.argmax(Y))
Y_global_max    = Y[global_max_ind]

all_idx         = list(range(total_samp))
all_idx_wo_max  = [i for i in all_idx if i != global_max_ind]

# Common split for benchmarking
splits_filepath = os.path.join(from_dir, "common_splits_1.pkl") # pkl file from the common_splits.py
with open(splits_filepath, "rb") as f:
    common_splits = pickle.load(f) 

# Loop around the splits
print(f"Start GP-PAM for {OUTER_LOOP * INNER_LOOP * len(common_splits)} runs…")
all_acq_hist, res_arr, all_results = [], [], []

for j in range(OUTER_LOOP):
    t_outer = time.time()
    for i in range(INNER_LOOP):
        for train_ind, test_ind in common_splits:
            res = PAM_regression_GP(kappa=KAPPA,
                                    save_csv=True,
                                    verbose=True,
                                    init_train_ind=train_ind,
                                    init_test_ind=test_ind,
                                    title=f"{DATASET_NAME}_LCB_run")
            all_results.append(res)
            res_arr.append(res[:-1])          
            all_acq_hist.append(res[-1]) 

    # Saving results
    summary_df = pd.DataFrame(res_arr, columns=[
        "file", "N_exp", "mean_wo_init", "std_wo_init",
        "mean_w_init", "std_w_init", "mean_init", "std_init", "run_min"
    ])
    utils.save_csv(summary_df, title=f"{DATASET_NAME}_summary")

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    with (to_dir / f"{DATASET_NAME}_LCB_acqHist_{ts}.pkl").open("wb") as fh:
        pickle.dump(all_acq_hist, fh)

    hrs = (time.time() - t_outer) / 3600
    print(f"Outer loop finished in {hrs:.2f} h — results saved.")
