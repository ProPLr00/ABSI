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
from scipy.stats import pearsonr, rankdata  
from sklearn.metrics import r2_score, mean_squared_error, ndcg_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from skopt.acquisition import gaussian_ei, gaussian_lcb
import argparse
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.base import clone   


import utils    

warnings.filterwarnings("ignore")
np.random.seed(3)
np.set_printoptions(precision=5)

# ── Parameters ─────────────────────────────────────────────────────────
name  = "CrTeNW_data"
UF    = "GP"
outer_loop= 2

root_dir = Path.cwd()
from_dir = root_dir / "data"
to_dir   = root_dir / "results"
to_dir.mkdir(parents=True, exist_ok=True)

class _SkoptDummy:
    """Holds vectorized mean/std; ignores X and returns (-mean, std) when asked."""
    def __init__(self, mean, std, negate_mean=True):
        self._mean = -mean if negate_mean else mean
        self._std  = std
    def predict(self, X, return_std=False):
        if return_std:  # X is unused; shape doesn't matter as long as len matches
            return self._mean, self._std
        return self._mean

def safe_pearson(y_true, y_pred):
    try:
        return pearsonr(y_true, y_pred)
    except Exception:
        return (np.nan, 1.0)

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

def _ensemble_mu_sigma(pred_matrix):
    """
    pred_matrix: shape (n_models, n_samples) or (n_samples,)
    returns: (mu, sigma), both shape (n_samples,)
    """
    P = np.asarray(pred_matrix, dtype=np.float64)
    if P.ndim == 1:   # allow 1D input
        P = P[None, :]
    n_models = P.shape[0]
    mu = P.mean(axis=0)
    if n_models == 1:
        sigma = np.full_like(mu, 1e-12)   # tiny floor avoids zero-acq ties
    else:
        sigma = P.std(axis=0, ddof=1)
        sigma = np.maximum(sigma, 1e-12)
    return mu, sigma

def PAM_regression_SVMEns(kappa=1.44, *, save_csv=False, verbose=False,
                          M=25,
                          tuned_parameters=None,
                          inner_nsplits=10,
                          init_train_ind=None, init_test_ind=None,
                          title="svm_run", to_break=True, batch=1,
                          seed=42, n_jobs=6):

    rng = np.random.RandomState(seed)
    random.seed(seed); np.random.seed(seed)

    if tuned_parameters is None:
        tuned_parameters = dict(
            reg__kernel=['rbf'],
            reg__tol=[1e-3, 1e-2, 1e-1],
            reg__C=[0.9, 1.0, 1.1],
            reg__epsilon=[0.0, 0.1, 0.2],
            reg__gamma=[1e-3, 1e-2, 1e-1, 1/6],
        )

    train_ind = sorted(init_train_ind.copy())
    test_ind  = sorted(init_test_ind.copy())
    results, acq_history = [], []
    start = time.time(); loop_count = 0; j = 0

    while test_ind:
        X_tr_raw = X_raw[train_ind].astype(np.float64, copy=False)
        X_te_raw = X_raw[test_ind].astype(np.float64,  copy=False)
        y_train  = Y[train_ind].astype(np.float64,    copy=False)
        y_test   = Y[test_ind].astype(np.float64,     copy=False)
        last_max = float(np.max(y_train))

        # per-iteration hyperparameter search
        n_tr = len(train_ind)
        cv_splits = max(2, min(inner_nsplits, n_tr))
        inner_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=j)

        svr_rbf = Pipeline([
            ('sc', StandardScaler()),
            ('reg', SVR())
        ])

        gs = GridSearchCV(
            estimator=svr_rbf,
            param_grid=tuned_parameters,
            cv=inner_cv,
            scoring='r2',
            verbose=int(bool(verbose)),
            n_jobs=n_jobs
        )
        gs.fit(X_tr_raw, y_train)
        best_pipe = gs.best_estimator_ 

        # bootstrap ensemble for μ, σ using the best hyperparams 
        preds = []
        for m in range(M):
            boot_idx = rng.choice(n_tr, size=n_tr, replace=True)
            model_m = clone(best_pipe)
            model_m.fit(X_tr_raw[boot_idx], y_train[boot_idx])
            preds.append(model_m.predict(X_te_raw))

        pred_matrix = np.vstack(preds)                
        mu, sigma   = _ensemble_mu_sigma(pred_matrix) 
        sigma = np.maximum(sigma, 1e-12)              

        # acquisition
        f_best = float(np.max(y_train))
        dummy  = _SkoptDummy(mean=mu, std=sigma)
        acq_vals = gaussian_lcb(mu.reshape(-1, 1), dummy, kappa=kappa, return_grad=False)
        acq_vals = np.asarray(acq_vals).ravel()
        acq_history.append(acq_vals)

        order      = np.argsort(acq_vals)           
        best_local = order[:batch]
        next_idx   = [test_ind[i] for i in best_local]

        # diagnostics on ensemble mean
        r2   = r2_score(y_test, mu)
        mse  = mean_squared_error(y_test, mu)
        pear, p_val = safe_pearson(y_test, mu)
        ndcg_k = ndcg_at_k(y_test, mu, k=min(batch, y_test.size))

        results.append([len(train_ind), next_idx, mu[best_local],
                        y_test[best_local], r2, mse, pear, p_val, ndcg_k,
                        acq_vals[best_local]])

        loop_count += 1; j += batch
        if verbose:
            print(f"{len(train_ind):3d} → add {next_idx}, y* = {y_test[best_local]}")
            print(loop_count, '->', j,
                  f"Train Size = {len(train_ind)}",
                  ', best_next_ind =', next_idx,
                  ', train_max =', f"{last_max:.6f}",
                  ', r2 =', f"{r2:.4f}",
                  ', LCB_val =', acq_vals[best_local].tolist(),
                  ", mu_pred =", mu[best_local].tolist(),
                  ", sigma_pred =", sigma[best_local].tolist())

        train_ind.extend(next_idx)
        test_ind = [ix for ix in test_ind if ix not in next_idx]
        if to_break and (Y[next_idx] == Y_global_max).any():
            break

    run_min = (time.time() - start) / 60
    saved = "-"
    if save_csv:
        cols = ['sample_size','pred_ind','best_pred_result','y_true',
                'r2','mse','pearson','p_value','ndcg','acq_value']
        saved = utils.save_csv_subfolder_UF(
            pd.DataFrame(results, columns=cols),
            title=title, UF="SVMEns", name=name
        )

    return [saved, len(train_ind), np.nan, np.nan,
            np.mean(Y[train_ind]), np.std(Y[train_ind]),
            np.mean(Y[init_train_ind]), np.std(Y[init_train_ind]),
            run_min, acq_history]

def PAM_regression_RFEns(kappa=1.44, *, save_csv=False, verbose=False,
                         tuned_parameters=None, inner_nsplits=10,
                         B=25,                          
                         n_estimators=500, max_depth=None, min_samples_leaf=1,
                         init_train_ind=None, init_test_ind=None,
                         title="rf_run", to_break=True, batch=1,
                         seed=42, n_jobs=6):

    random.seed(seed); np.random.seed(seed)

    # Default RF grid
    if tuned_parameters is None:
        tuned_parameters = dict(
            n_estimators=[300, 500, 800],
            max_depth=[None, 6, 12],
            min_samples_leaf=[1, 2, 4],
            max_features=["sqrt", 0.6, 1.0],
            bootstrap=[True],
        )

    train_ind = sorted(init_train_ind.copy())
    test_ind  = sorted(init_test_ind.copy())
    results, acq_history = [], []
    start = time.time(); loop_count = 0; j = 0

    while test_ind:
        X_tr_raw = X_raw[train_ind].astype(np.float64, copy=False)
        X_te_raw = X_raw[test_ind].astype(np.float64,  copy=False)
        y_train  = Y[train_ind].astype(np.float64,    copy=False)
        y_test   = Y[test_ind].astype(np.float64,     copy=False)
        last_max = float(np.max(y_train))

        # hyperparameter search with CV
        n_tr = len(train_ind)
        cv_splits = max(2, min(inner_nsplits, n_tr))
        inner_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=j)

        base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=seed
        )

        rf_cv = GridSearchCV(
            estimator=base_rf,
            param_grid=tuned_parameters,
            cv=inner_cv,
            scoring='r2',
            n_jobs=n_jobs,
            verbose=int(bool(verbose))
        )
        rf_cv.fit(X_tr_raw, y_train)
        best_rf = rf_cv.best_estimator_ 

        if verbose:
            print("Best RF params:", rf_cv.best_params_)

        # uncertainty via ensemble of forests (B) or per-tree
        if B >= 2:
            preds = []
            for b in range(B):
                boot_idx = np.random.RandomState(seed + b).choice(n_tr, size=n_tr, replace=True)
                rf_b = clone(best_rf)
                rf_b.set_params(random_state=seed + 1337 + b)
                rf_b.fit(X_tr_raw[boot_idx], y_train[boot_idx])
                preds.append(rf_b.predict(X_te_raw))
            pred_matrix = np.vstack(preds)
            mu, sigma   = _ensemble_mu_sigma(pred_matrix)
        else:
            # Fast path: per-tree predictive variance from a single forest
            rf_1 = clone(best_rf)
            rf_1.set_params(random_state=seed)
            rf_1.fit(X_tr_raw, y_train)
            tree_preds = np.vstack([est.predict(X_te_raw) for est in rf_1.estimators_])
            mu, sigma  = _ensemble_mu_sigma(tree_preds)

        sigma = np.maximum(sigma, 1e-12)

        # acquisition
        f_best = float(np.max(y_train))
        dummy  = _SkoptDummy(mean=mu, std=sigma)
        acq_vals = gaussian_lcb(mu.reshape(-1,1), dummy, kappa=kappa, return_grad=False)
        acq_vals = np.asarray(acq_vals).ravel()
        acq_history.append(acq_vals)

        order = np.argsort(acq_vals)   
        best_local = order[:batch]
        next_idx   = [test_ind[i] for i in best_local]

        r2   = r2_score(y_test, mu)
        mse  = mean_squared_error(y_test, mu)
        pear, p_val = safe_pearson(y_test, mu)
        ndcg_k = ndcg_at_k(y_test, mu, k=min(batch, y_test.size))

        results.append([len(train_ind), next_idx, mu[best_local],
                        y_test[best_local], r2, mse, pear, p_val, ndcg_k,
                        acq_vals[best_local]])

        loop_count += 1; j += batch
        if verbose:
            print(f"{len(train_ind):3d} → add {next_idx}, y* = {y_test[best_local]}")
            print(loop_count, '->', j,
                  f"Train Size = {len(train_ind)}",
                  ', best_next_ind =', next_idx,
                  ', train_max =', f"{last_max:.6f}",
                  ', r2 =', f"{r2:.4f}",
                  ', LCB_val =', acq_vals[best_local].tolist(),
                  ", mu_pred =", mu[best_local].tolist(),
                  ", sigma_pred =", sigma[best_local].tolist())

        train_ind.extend(next_idx)
        test_ind = [ix for ix in test_ind if ix not in next_idx]
        if to_break and (Y[next_idx] == Y_global_max).any():
            break

    run_min = (time.time() - start) / 60
    saved = "-"
    if save_csv:
        cols = ['sample_size','pred_ind','best_pred_result','y_true','r2','mse','pearson','p_value','ndcg','acq_value']
        saved = utils.save_csv_subfolder_UF(pd.DataFrame(results, columns=cols), title=title, UF="RFEns", name=name)

    return [saved, len(train_ind), np.nan, np.nan,
            np.mean(Y[train_ind]), np.std(Y[train_ind]),
            np.mean(Y[init_train_ind]), np.std(Y[init_train_ind]),
            run_min, acq_history]

def PAM_regression_XGBEns(
    acq_func="LCB",                # "LCB" or "EI"
    kappa=1.44, xi=0.01,           # EI uses xi; LCB uses kappa
    *, save_csv=False, verbose=False,
    M=3,                          # ensemble size
    tuned_parameters=None,         # grid for GridSearchCV
    inner_nsplits=10,              # CV for the grid search
    init_train_ind=None, init_test_ind=None,
    title="xgb_run", to_break=True, batch=1, seed=42, n_jobs=6
):
    rng = np.random.RandomState(seed)
    random.seed(seed); np.random.seed(seed)

    if tuned_parameters is None:
        tuned_parameters = dict(
            learning_rate=[0.01],
            n_estimators=[300, 500, 700],
            colsample_bylevel=[0.5, 0.7, 0.9],
            gamma=[0, 0.2],
            max_depth=[3, 7, 11],
            reg_lambda=[0.1, 1, 10],
            subsample=[0.4, 0.7, 1.0],
        )

    train_ind = sorted(init_train_ind.copy())
    test_ind  = sorted(init_test_ind.copy())
    results, acq_history = [], []
    start = time.time(); loop_count = 0; j = 0

    while test_ind:
        # scale per-iteration on TRAIN only
        X_tr_raw = X_raw[train_ind].astype(np.float64, copy=False)
        X_te_raw = X_raw[test_ind].astype(np.float64,  copy=False)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_tr_raw)
        X_test  = scaler.transform(X_te_raw)
        y_train = Y[train_ind].astype(np.float64, copy=False)
        y_test  = Y[test_ind].astype(np.float64,  copy=False)
        last_max = float(np.max(y_train))

        # 1) Per-iteration hyperparameter search (same as PAM_regression)
        inner_cv = KFold(n_splits=inner_nsplits, shuffle=True, random_state=j)
        base = xgb.XGBRegressor(objective="reg:linear", min_child_weight=1, **{'tree_method': 'exact'},
                                silent=True, n_jobs=n_jobs, random_state=3, seed=3)
        gs = GridSearchCV(base, tuned_parameters, cv=inner_cv, scoring='r2', verbose=0, n_jobs=n_jobs)
        gs.fit(X_train, y_train)
        best_params = gs.best_estimator_.get_params()

        # 2) Bootstrap + seed-offset ensemble
        preds = []
        for m in range(M):
            # bootstrap indices
            boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train[boot_idx], y_train[boot_idx], verbose=False)
            preds.append(model.predict(X_test))
        pred_matrix = np.vstack(preds)                
        mu, sigma   = _ensemble_mu_sigma(pred_matrix) 

        # 3) Acquisition via skopt (pick *smallest* acq value)
        f_best = float(np.max(y_train))
        dummy  = _SkoptDummy(mean=mu, std=sigma)  
        X_cand = mu.reshape(-1, 1)               

        if acq_func.upper() == "LCB":
            acq_vals = gaussian_lcb(X_cand, dummy, kappa=kappa, return_grad=False)
        elif acq_func.upper() == "EI":
            acq_vals = gaussian_ei(X_cand, dummy, y_opt=-f_best, xi=xi, return_grad=False)
        else:
            raise ValueError("acq_func must be 'LCB' or 'EI'.")

        acq_vals = np.asarray(acq_vals).ravel()
        acq_history.append(acq_vals)
        order = np.lexsort((np.arange(acq_vals.size), acq_vals)) 
        best_local = order[:batch]
        next_idx   = [test_ind[i] for i in best_local]

        # 4) Diagnostics (computed on ensemble mean)
        r2   = r2_score(y_test, mu)
        mse  = mean_squared_error(y_test, mu)
        pear, p_val = safe_pearson(y_test, mu)
        ndcg_k = ndcg_at_k(y_test, mu, k=batch)
        results.append([len(train_ind), next_idx, mu[best_local],
                        y_test[best_local], r2, mse, pear, p_val, ndcg_k,
                        acq_vals[best_local]])

        if verbose:
            print(f"{len(train_ind):3d} → add {next_idx}, y* = {y_test[best_local]}")
            print(loop_count+1, '->', j+batch,
                  f"Train Size = {len(train_ind)}",
                  ', best_next_ind =', next_idx,
                  ', train_max =', f"{last_max:.6f}",
                  ', r2 =', f"{r2:.4f}",
                  f", {acq_func}_val =", acq_vals[best_local].tolist(),
                  ", mu_pred =", mu[best_local].tolist(),
                  ", sigma_pred =", sigma[best_local].tolist())

        # 5) Pool updates + early stop
        train_ind.extend(next_idx)
        test_ind = [ix for ix in test_ind if ix not in next_idx]
        j += batch; loop_count += 1
        if to_break and (Y[next_idx] == Y_global_max).any():
            break

    run_min = (time.time() - start) / 60
    saved = "-"
    if save_csv:
        cols = ['sample_size', 'pred_ind', 'best_pred_result', 'y_true','r2','mse','pearson','p_value','ndcg','acq_value']
        saved = utils.save_csv_subfolder_UF(pd.DataFrame(results, columns=cols), title=title, UF="XGBEns", name=name)

    return [saved, len(train_ind), np.nan, np.nan,
            np.mean(Y[train_ind]), np.std(Y[train_ind]),
            np.mean(Y[init_train_ind]), np.std(Y[init_train_ind]),
            run_min, acq_history]

def PAM_regression_GP(kappa=1.44, *, save_csv=False, verbose=False,
                      m_restarts=10, init_train_ind=None, init_test_ind=None,
                      title="gp_run", to_break=True, batch=1, seed=42):

    random.seed(seed)
    np.random.seed(seed)

    train_ind = sorted(init_train_ind.copy())
    test_ind  = sorted(init_test_ind.copy())
    results, acq_history = [], []
    loop_count, j = 0, 0
    start = time.time()
    while test_ind:
        X_tr_raw = X_raw[train_ind].astype(np.float64, copy=False)
        X_te_raw = X_raw[test_ind].astype(np.float64,  copy=False)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_tr_raw)
        X_test  = scaler.transform(X_te_raw)
        y_train = Y[train_ind].astype(np.float64, copy=False)
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
        f_best = float(np.max(y_train))
        dummy  = _SkoptDummy(mean=mu, std=sigma)

        acq_vals = gaussian_lcb(mu.reshape(-1,1), dummy, kappa=kappa, return_grad=False)
        # acq_vals = gaussian_ei(mu.reshape(-1,1), dummy, y_opt=-f_best, xi=0.01, return_grad=False)

        acq_vals = np.asarray(acq_vals).ravel()
        acq_history.append(acq_vals)
        order = np.lexsort((np.arange(acq_vals.size), acq_vals))  # argmin
        best_local = order[:batch]

        next_idx   = [test_ind[i] for i in best_local]

        # Diagnostics
        r2   = r2_score(y_test, mu)
        mse  = mean_squared_error(y_test, mu)
        pear, p_val = safe_pearson(y_test, mu)    
        ndcg_k = ndcg_at_k(y_test, mu, k=batch)

        results.append([len(train_ind), next_idx, mu[best_local],
                        y_test[best_local], r2, mse, pear, p_val, ndcg_k,
                        acq_vals[best_local]])
        best_y_true_list = np.round(y_test[best_local], 6).tolist()
        lcb_vals_list    = acq_vals[best_local].tolist()

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
        saved = utils.save_csv_subfolder_UF(pd.DataFrame(results, columns=cols), title=title, UF="GP", name=name)

    return [saved, len(train_ind), np.nan, np.nan,
            np.mean(Y[train_ind]), np.std(Y[train_ind]),
            np.mean(Y[init_train_ind]), np.std(Y[init_train_ind]),
            run_min, acq_history]

# Data loading 
df_cleaned, X_raw, Y, clean_feature_list, clean_result_col = utils.load_and_clean_data(
    name, feature_col_num=0, target="length"
)

total_samp     = X_raw.shape[0]
global_max_ind = int(np.argmax(Y))
Y_global_max   = Y[global_max_ind]

all_idx        = list(range(total_samp))
all_idx_wo_max = [i for i in all_idx if i != global_max_ind]

# Loops 
inner_loop = 1      # Replace common_split for randomly select initial seeding point

def get_runner(model):
    m = model.upper()
    if m == "GP":
        return "GP", PAM_regression_GP, dict(kappa=1.44, m_restarts=10)
    if m == "RFENS":
        # RF: bag B forests for σ parity with SVM/XGB bootstraps
        return "RFEns", PAM_regression_RFEns, dict(kappa=1.44, inner_nsplits=10, B=25)
    if m == "SVMENS":
        # SVM: bootstrap M pipelines
        return "SVMEns", PAM_regression_SVMEns, dict(kappa=1.44, inner_nsplits=10, M=25)
    if m == "XGBENS":
        # XGB: bootstrap M models
        return "XGBEns", PAM_regression_XGBEns, dict(kappa=1.44, inner_nsplits=10, M=25)
    raise ValueError("model must be one of: GP, RFEns, XGBEns, SVMEns")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["GP", "RFEns", "XGBEns", "SVMEns"], required=True)
    parser.add_argument("--outer_loop", type=int, default=2)
    parser.add_argument("--inner_loop", type=int, default=1)
    args = parser.parse_args()

    UF, runner, kwargs = get_runner(args.model)
    outer_loop = args.outer_loop
    inner_loop = args.inner_loop

    # Reuse same splits for fairness
    splits_filepath = from_dir / "common_splits_1.pkl"
    with splits_filepath.open("rb") as f:
        common_splits = pickle.load(f)

    n_runs = outer_loop * inner_loop * len(common_splits)
    print(f"\n=== Start {UF}-PAM for {n_runs} runs… ===")

    all_acq_hist, res_arr, all_results = [], [], []
    t_outer = time.time()
    seed_base = 42

    for i_outer in range(outer_loop):
        for i_inner in range(inner_loop):
            for i_split, (train_ind, test_ind) in enumerate(common_splits):
                run_seed = seed_base + i_outer*10_000 + i_inner*1_000 + i_split
                try:
                    res = runner(
                        save_csv=True,
                        verbose=True,
                        init_train_ind=train_ind,
                        init_test_ind=test_ind,
                        title=f"PAM_{UF}_{name}_",
                        seed=run_seed,
                        **kwargs
                    )
                except Exception as e:
                    print(f"[WARN] {UF} run skipped (outer={i_outer}, inner={i_inner}, split={i_split}): {e}")
                    continue

                all_results.append(res)
                res_arr.append(res[:-1])     # everything except acq history
                all_acq_hist.append(res[-1]) # acq history

    subfolder_path = to_dir / f"{UF}_{name}"
    subfolder_path.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(
        data=res_arr,
        columns=[
            'file-name','num_experiments','mean_y_wo_init','std_y_wo_init',
            'mean_y_w_init','std_y_w_init','mean_y_only_init','std_y_only_init','run_time'
        ]
    )
    utils.save_csv_subfolder_UF(summary_df, title=f"Summary_PAM_{UF}_{name}_", UF=UF, name=name)

    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    with (subfolder_path / f"Acqhist_PAM_{UF}_{name}_{stamp}.pkl").open("wb") as fh:
        pickle.dump(all_acq_hist, fh)

    hrs = (time.time() - t_outer) / 3600
    print(f"{UF}: finished in {hrs:.2f} h — results saved to {subfolder_path}.")

