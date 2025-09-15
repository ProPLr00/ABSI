import numpy as np
import pandas as pd
import time
from pathlib import Path
import os

import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,RationalQuadratic

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr, rankdata
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GridSearchCV, KFold

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import utils

# Name of the data sheet in .csv form under from_dir
name = "CrTeNW_data"

root_dir = str(Path(os.getcwd()))
from_dir = root_dir + "/data/"
to_dir = root_dir + "/results/"
save = True

# Load the data, please specify if you are using a different dir than from_dir with data_dir = your_dir
df_cleaned, X, Y, clean_feature_list, clean_result_col = utils.load_and_clean_data(name, feature_col_num=0, target="length")

def test(y_pred, y_true, k=None, verbose=False):
    """
    Evaluate the performance of a regression model using R², MSE, Pearson correlation, p-value, and nDCG.
    Also tracks the index of the data point with the largest loss.

    Parameters:
        y_pred : array-like
            Predicted target values.
        y_true : array-like
            Ground truth target values.
        k : int, optional (default=10)
            The number of top results to consider for nDCG evaluation.
        verbose : bool, optional (default=False)
            If True, prints rankings, actual values, predicted values, and largest loss details.

    Returns:
        metrics_array : np.array
            [r², mse, pearson, pearson_p_value, ndcg_score].
        max_loss_index : int
            Index of the data point with the largest loss.
    """
    # Ensure inputs are NumPy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape.")
    
    if len(y_pred) == 0:
        raise ValueError("Input arrays must not be empty.")
    
    # R² and MSE
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    
    # Pearson correlation and p-value
    pear, p_value = pearsonr(y_true, y_pred)
    
    # Generate rankings with ties
    true_ranking = rankdata(-y_true, method="min")  # Higher values get lower ranks
    pred_ranking = rankdata(-y_pred, method="min")  # Higher predicted values get lower ranks

    # Compute nDCG using relevance scores
    ndcg = ndcg_score([y_true], [y_pred], k=k)
    
    # Identify the index of the largest loss
    losses = np.abs(y_pred - y_true)
    max_loss_index = np.argmax(losses)
    
    if verbose:
        print("True Ranking:", true_ranking)
        print("Predicted Ranking:", pred_ranking)
        print(f"Largest Loss: {losses[max_loss_index]} at Index: {max_loss_index}")
        print("\nActual Values:")
        print(y_true)
        print("\nPredicted Values:")
        print(y_pred)
    
    return np.array([r2, mse, pear, p_value, ndcg]), max_loss_index

def compute_mean_std(X):   
    feature_list = clean_feature_list
    print('\n\n>>Feature stats:')
    for i in range(len(feature_list)):
        arr = X[feature_list[i]]
        mean =  np.mean(arr)
        std = np.std(arr)
        print(feature_list[i],':   mean= ',mean,' std= ',std)

# Plot the correlation matrix
title = name
utils.plot_correlation_matrix(X, title, clean_feature_list,toSaveFig=save)

# Adjust running setup
verbose=False
n_jobs = 6
save_csv = True

# cross validation setup
Ntrials = 10
outter_nsplit = 5
inner_nsplits = 5

print('start  ',str(Ntrials),' trials...')
tot_count = Ntrials * outter_nsplit

# Results store with an additional column for nDCG
svr_mat = np.zeros((tot_count, 5))
xgb_mat = np.zeros((tot_count, 5))
mlp_mat = np.zeros((tot_count, 5))
gpr_mat = np.zeros((tot_count, 5))

for i in range(Ntrials):
    init_time = time.time()
    train_index = []  
    test_index = []  

    outer_cv = KFold(n_splits=outter_nsplit, shuffle=True, random_state=i+9)
    for train_ind, test_ind in outer_cv.split(X, Y):
        train_index.append(train_ind.tolist())
        test_index.append(test_ind.tolist())

    #Outer nsplits
    for j in range(outter_nsplit):
        count = i * outter_nsplit + j
        iter_start = time.time()
        print(str(count), "  / ",str(tot_count))
        X_train = X[train_index[j]]
        Y_train = Y[train_index[j]]

        X_test = X[test_index[j]]
        Y_test = Y[test_index[j]]

        inner_cv = KFold(n_splits=inner_nsplits, shuffle=True, random_state=j)  


        # gpr
        kernel =  Matern(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), nu=1.5) +\
                RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-05, 100000.0), alpha_bounds=(1e-05, 100000.0)) 
        gpr_reg = Pipeline([            
                ('sc', StandardScaler()), 
                ('reg',  GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10))
                ])

        gpr_reg.fit(X_train, Y_train)
        y_pred = gpr_reg.predict(X_test)
        gpr_metrics, _ = test(y_pred, Y_test)
        gpr_mat[count] = gpr_metrics

        # SVR - rbf
        svr_rbf = Pipeline([            
                ('sc', StandardScaler()), 
                ('reg',  SVR())
                ])
        tuned_parameters = dict(reg__kernel=['rbf'],
                                reg__tol= [1e-3,1e-2,1e-1],
                                reg__C=[0.9,1,1.1],
                                reg__epsilon=[0,0.1,0.2],
                                reg__gamma=[1e-3,1e-2,1e-1,1/6]
                              )
        svr_cv = GridSearchCV(svr_rbf,tuned_parameters, cv=inner_cv, scoring='r2',verbose=verbose,n_jobs=n_jobs)
        svr_cv.fit(X_train, Y_train)
        y_pred = svr_cv.predict(X_test)
        svr_metrics, _ = test(y_pred, Y_test)
        svr_mat[count] = svr_metrics

        # GradientBoost
        tuned_parameters = dict(objective=["reg:linear"],
                            learning_rate=[0.01],
                            n_estimators=[300,500,700], #100,,300,400,500
                            colsample_bylevel = [0.5,0.7,0.9],
                          gamma=[0,0.2], #0,0.1,0.2,0.3,0.4
                          max_depth =[3,7,11], # [3,7,11]]
                          reg_lambda = [0.1,1,10], #[0.1,1,10]
                         # reg_alpha = [1],
                           subsample=[0.4,0.7,1])

        xgb_reg = xgb.XGBRegressor(min_child_weight=1,**{'tree_method':'exact'},
                                 silent=True,n_jobs=n_jobs,random_state=3,seed=3)

        xgb_cv = GridSearchCV(xgb_reg,tuned_parameters, cv=inner_cv,scoring='r2',verbose=verbose,n_jobs=n_jobs)
        xgb_cv.fit(X_train, Y_train)
        y_pred = xgb_cv.predict(X_test)
        xgb_metrics, _ = test(y_pred, Y_test)
        xgb_mat[count] = xgb_metrics

        # MLP
        mlp_clf = Pipeline([            
                ('sc', StandardScaler()), 
                ('reg',  MLPRegressor(random_state=3))
                ])
        tuned_parameters = dict(reg__hidden_layer_sizes=[[20,20], [10,10,10]],
                            reg__alpha=[1e-4, 1e-3, 1e-2], 
                            reg__solver= ['lbfgs'],
                            reg__max_iter=[6000])
        mlp_cv = GridSearchCV(mlp_clf, tuned_parameters, cv=inner_cv,scoring='r2',verbose=verbose,n_jobs=n_jobs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            mlp_cv.fit(X_train, Y_train)
        y_pred = mlp_cv.predict(X_test)
        mlp_metrics, _ = test(y_pred, Y_test)
        mlp_mat[count] = mlp_metrics

        elapsed = time.time() - iter_start
        print(f"In Ntrials {i+1} - Iteration {j+1} took {elapsed:.2f} seconds")
        if verbose:
            print('svr - ',svr_mat[count])
            print('gpr - ',gpr_mat[count])
            print('mlp - ',mlp_mat[count])
            print('xgb -',xgb_mat[count])
    print((time.time()-init_time)/60, ' min')

# Update DataFrames
mlp_results = pd.DataFrame(data=mlp_mat, columns=['r2', 'mse', 'pear', 'pear_p_val', 'ndcg'])
gpr_results = pd.DataFrame(data=gpr_mat, columns=['r2', 'mse', 'pear', 'pear_p_val', 'ndcg'])
xgb_results = pd.DataFrame(data=xgb_mat, columns=['r2', 'mse', 'pear', 'pear_p_val', 'ndcg'])
svr_results = pd.DataFrame(data=svr_mat, columns=['r2', 'mse', 'pear', 'pear_p_val', 'ndcg'])

if(save_csv):
    utils.save_csv(gpr_results, title='[model_selection_reg]gpr_results')
    utils.save_csv(mlp_results, title='[model_selection_reg]mlp_results')
    utils.save_csv(xgb_results, title='[model_selection_reg]xgb_results')
    utils.save_csv(svr_results, title='[model_selection_reg]svr_results')

print('end ',str(Ntrials),' trials')

np.random.seed(44)

# Print summary
print('->>>XGBoost_mean : \n', xgb_results.mean(axis=0), '\n  std = \n', xgb_results.std(axis=0))
print('->>>SVR_mean : \n', svr_results.mean(axis=0), ' \n std =\n', svr_results.std(axis=0))
print('->>>MLP_mean : \n', mlp_results.mean(axis=0), ' \n std =\n', mlp_results.std(axis=0))
print('->>>GPR_mean : \n', gpr_results.mean(axis=0), ' \n std =\n', gpr_results.std(axis=0))

labels = ['XGBoost-R', 'MLP-R', 'SVM-R', 'GP-R']
n_results = outter_nsplit * Ntrials   # rows per metric

r2_results   = pd.DataFrame({'XGBoost-R': xgb_results['r2'],
                             'MLP-R'    : mlp_results['r2'],
                             'SVM-R'    : svr_results['r2'],
                             'GP-R'     : gpr_results['r2']})

mse_results  = pd.DataFrame({'XGBoost-R': xgb_results['mse'],
                             'MLP-R'    : mlp_results['mse'],
                             'SVM-R'    : svr_results['mse'],
                             'GP-R'     : gpr_results['mse']})

pear_results = pd.DataFrame({'XGBoost-R': xgb_results['pear'],
                             'MLP-R'    : mlp_results['pear'],
                             'SVM-R'    : svr_results['pear'],
                             'GP-R'     : gpr_results['pear']})

ndcg_results = pd.DataFrame({'XGBoost-R': xgb_results['ndcg'],
                             'MLP-R'    : mlp_results['ndcg'],
                             'SVM-R'    : svr_results['ndcg'],
                             'GP-R'     : gpr_results['ndcg']})

data  = [r2_results, mse_results, pear_results, ndcg_results]
ylabels  = [r'$R^2$', 'MSE', r'$r$', 'nDCG']

plt.rcdefaults()

# Saving Metrics
metrics = {
    r'$R^2$':   r2_results,
    'MSE':      mse_results,
    r'$r$':     pear_results,
    'nDCG':     ndcg_results
}
models = ['XGBoost-R', 'MLP-R', 'SVM-R', 'GP-R']
colors = {
    'XGBoost-R': 'blue',
    'MLP-R'    : 'green',
    'SVM-R'    : 'orange',
    'GP-R'     : 'red'
}
xlimits = {
    r'$R^2$': (0, 1),
    'MSE'   : (0, 100),
    r'$r$'  : (0, 1),
    'nDCG'  : (0.7, 1)
}

# Plotting in 2x2 pattern
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axes = axes.flatten()

# Loop over each metric with one subplot each
for ax, (xlabel, df) in zip(axes, metrics.items()):
    # Prepare data in model order
    data = [df[model] for model in models]
    # Draw horizontal boxplot with colored patches
    bp = ax.boxplot(data,
                    labels=models,
                    vert=False,
                    patch_artist=True,
                    medianprops={'color': 'black'})
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(colors[model])
        patch.set_edgecolor('black')
    # Axis limits, labels, title
    ax.set_xlim(xlimits[xlabel])
    ax.set_xlabel(xlabel)
    ax.set_yticklabels(models)
    ax.set_title(f'{xlabel} distribution')

# Overall title and show
fig.suptitle('Comparison of Model Metrics', fontsize=16, y=1.02)
filename = os.path.join(to_dir, "model_metrics_boxplots.png")
fig.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Figure written to {filename}")