import os, time, random, datetime, pickle, warnings
from pathlib import Path

# For threads and seed control from os
os.environ["PYTHONHASHSEED"]     = "42"
os.environ["OMP_NUM_THREADS"]       = "12"   
os.environ["MKL_NUM_THREADS"]       = "12"   
os.environ["OPENBLAS_NUM_THREADS"]  = "12"   
os.environ["NUMEXPR_NUM_THREADS"]   = "12"   
os.environ["VECLIB_MAXIMUM_THREADS"]= "12"   

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, ndcg_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

import utils    

warnings.filterwarnings("ignore")
np.random.seed(3)
np.set_printoptions(precision=5)

# ── Parameters ─────────────────────────────────────────────────────────
name  = "CrTeNW_data"
UF    = "GP"
outer_loop= 10

root_dir = Path.cwd()
from_dir = root_dir / "data"
to_dir   = root_dir / "results"
to_dir.mkdir(parents=True, exist_ok=True)

def ndcg_at_k(y_true, y_score, k=None):
    """
    Safe NDCG:
    - reshapes to (1, n) as sklearn expects
    - shifts y_true if it contains negatives (sklearn warns score may not be in [0,1])
    - caps k to len(y_true)
    """
    y_true  = np.asarray(y_true, dtype=float).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.size == 0:
        return np.nan
    if k is None or k > y_true.size:
        k = y_true.size
    # shift if negatives to keep the metric well-behaved
    mn = y_true.min()
    if mn < 0:
        y_true = y_true - mn
    try:
        return float(ndcg_score(y_true.reshape(1, -1),
                                y_score.reshape(1, -1),
                                k=int(k)))
    except Exception:
        return np.nan

# Main GP function
def PAM_regression_GP(kappa=1.44, *, save_csv=False, verbose=False,
                      m_restarts=10, init_train_ind=None, init_test_ind=None,
                      title="gp_run", to_break=True, batch=1, seed=42):

    # For fixing seed
    random.seed(seed)
    np.random.seed(seed)

    train_ind = sorted(init_train_ind.copy())
    test_ind  = sorted(init_test_ind.copy())
    results, acq_history = [], []
    loop_count, j = 0, 0
    start = time.time()
    while test_ind:
        X_train = X[train_ind].astype(np.float64, copy=False)
        y_train = Y[train_ind].astype(np.float64, copy=False)
        X_test  = X[test_ind].astype(np.float64,  copy=False)
        y_test  = Y[test_ind].astype(np.float64,  copy=False)
        last_max = float(np.max(y_train))
        # GP fit (Matern-5/2, ARD), reproducible optimizer restarts
        d = X_train.shape[1]
        kernel = (C(1.0, (1e-3, 1e3)) *
                  Matern(length_scale=np.ones(d),
                         length_scale_bounds=(1e-2, 1e2), nu=2.5)
                 ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1.0))
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       n_restarts_optimizer=m_restarts,
                                       normalize_y=True,
                                       random_state=seed)
        gpr.fit(X_train, y_train)
        mu, sigma = gpr.predict(X_test, return_std=True)

        # LCB acquisition
        acq = mu - kappa * sigma
        acq_history.append(acq)
        order = np.lexsort((np.arange(acq.size), acq))  
        best_local = order[:batch]
        next_idx   = [test_ind[i] for i in best_local]

        # Diagnostics
        r2   = r2_score(y_test, mu)
        mse  = mean_squared_error(y_test, mu)
        pear, p_val = pearsonr(y_test, mu)    
        ndcg_k = ndcg_at_k(y_test, mu, k=batch)

        results.append([len(train_ind), next_idx, mu[best_local],
                        y_test[best_local], r2, mse, pear, p_val, ndcg_k,
                        acq[best_local]])
        best_y_true_list = np.round(y_test[best_local], 6).tolist()
        lcb_vals_list    = acq[best_local].tolist()

        loop_count += 1
        j += batch
        
        if verbose:
            print(f"{len(train_ind):3d} → add {next_idx}, y* = {y_test[best_local]}")
            print(
                loop_count, '->', j,
                f"Train Size = {len(train_ind)}",
                ', best_next_ind =', next_idx,
                ', best_Y_true =', best_y_true_list,
                ', train_max =', f"{last_max:.6f}",
                ', r2 =', f"{r2:.4f}",
                ', LCB_val =', lcb_vals_list,
                ", mu_pred =", mu[best_local].tolist(), 
                ", sigma_pred =", sigma[best_local].tolist()
            )
            
        # Update pools deterministically
        train_ind.extend(next_idx)
        test_ind = [ix for ix in test_ind if ix not in next_idx]

        # Early stop is deterministic once selection is deterministic
        if to_break and (Y[next_idx] == Y_global_max).any():
            break
        

    run_min = (time.time() - start) / 60
    saved = "-"
    if save_csv:
        cols = ['sample_size', 'pred_ind', 'best_pred_result', 'y_true','r2', 'mse', 'pearson', 'p_value', 'ndcg', 'acq_value']
        saved = utils.save_csv_subfolder_UF(pd.DataFrame(results, columns=cols), title=title, UF=UF, name=name)

    return [saved, len(train_ind), np.nan, np.nan,
            np.mean(Y[train_ind]), np.std(Y[train_ind]),
            np.mean(Y[init_train_ind]), np.std(Y[init_train_ind]),
            run_min, acq_history]

# ── Data loading ───────────────────────────────────────────────────────
df_cleaned, X, Y, clean_feature_list, clean_result_col = utils.load_and_clean_data(
    name, feature_col_num=0, target="length"
)

scaler = StandardScaler()
X = scaler.fit_transform(X)

total_samp     = X.shape[0]
global_max_ind = int(np.argmax(Y))
Y_global_max   = Y[global_max_ind]

all_idx        = list(range(total_samp))
all_idx_wo_max = [i for i in all_idx if i != global_max_ind]

# ── Common splits ──────────────────────────────────────────────────────
splits_filepath = from_dir / "common_splits_1.pkl"
with splits_filepath.open("rb") as f:
    common_splits = pickle.load(f)

today = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
splits_out = from_dir / f"Common_splits_{UF}_{name}_{today}.pkl"
with splits_out.open('wb') as f:
    pickle.dump(common_splits, f)
print("Initial splits saved to:", splits_out)

# ── Loops ──────────────────────────────────────────────────────────────
inner_loop = 1      # Replace common_split for randomly select initial seeding point

print(f"Start GP-PAM for {outer_loop * inner_loop * len(common_splits)} runs…")
all_acq_hist, res_arr, all_results = [], [], []

for j in range(outer_loop):
    t_outer = time.time()
    for i in range(inner_loop):
        for train_ind, test_ind in common_splits:
            res = PAM_regression_GP(
                kappa=1.44,
                save_csv=True,
                verbose=True,
                init_train_ind=train_ind,
                init_test_ind=test_ind,
                title=f"PAM_{UF}_{name}_"
            )
            all_results.append(res)
            res_arr.append(res[:-1])       # 9 items for 9 columns
            all_acq_hist.append(res[-1])   # acquisition history

    # ── Save per-outer summary CSV into {to_dir}/{UF}_{name}/ ──────────
    subfolder_path = to_dir / f"{UF}_{name}"
    subfolder_path.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        data=res_arr,
        columns=[
            'file-name','num_experiments','mean_y_wo_init','std_y_wo_init',
            'mean_y_w_init','std_y_w_init','mean_y_only_init','std_y_only_init','run_time'
        ]
    )
    utils.save_csv_subfolder_UF(
        summary_df,
        title=f"Summary_PAM_{UF}_{name}_",
        UF=UF,
        name=name
    )

    # ── Save acquisition history alongside the summaries (same subfolder) ─
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    acq_history_path = subfolder_path / f"Acqhist_PAM_{UF}_{name}_{stamp}.pkl"
    with acq_history_path.open("wb") as fh:
        pickle.dump(all_acq_hist, fh)

    hrs = (time.time() - t_outer) / 3600
    print(f"Outer loop finished in {hrs:.2f} h — results saved to {subfolder_path}.")
