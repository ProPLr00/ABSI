import os
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

UF = 'EI'
name = 'CrTeNW_data'

root_dir = str(Path(os.getcwd()))
from_dir = root_dir + '/data/'
to_dir = root_dir + '/results/'

def update_title_w_date(title):
    now_time = datetime.datetime.now()
    today = str(now_time.year)+'_'+str(now_time.month)+'_'+str(now_time.day)
    return title + today

def format_title(to_dir, title, fileEtd):
    title = update_title_w_date(title)
    to_save_title = to_dir+title+fileEtd
    
    i=0
    while(os.path.exists(to_save_title)):
        to_save_title = to_dir+title+'_'+str(i)+fileEtd
        i= i+1
    return to_save_title

def format_title_subfolder_UF(to_dir: str, title: str, *, UF: str, name: str, file_ext: str = ".csv") -> str:
    """Build a unique save path under {to_dir}/{UF}_{name}/"""
    title2 = update_title_w_date(title)
    subfolder_path = os.path.join(to_dir, f"{UF}_{name}")
    os.makedirs(subfolder_path, exist_ok=True)

    filename = f"{title2}{file_ext}"
    full_path = os.path.join(subfolder_path, filename)

    i = 1
    while os.path.exists(full_path):
        filename = f"{title2}_{i}{file_ext}"
        full_path = os.path.join(subfolder_path, filename)
        i += 1

    return full_path

def set_fig_fonts(SMALL_SIZE=22, MEDIUM_SIZE=24,BIGGER_SIZE = 26):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def save_fig(fig, title):
    to_path = format_title(to_dir,title,'.png')
    fig.savefig(to_path ,dpi=1000,bbox_inches="tight",pad_inches=0)#, bbox_inches='tight', pad_inches=10
    print("Successfully saved to: ",to_path)
    return to_path

def plot_boxplots(data=None, ylabels = [], xmin =0.2, xmax= 1.025,toSaveFig = True, title='boxplot',
                       palette_colors=sns.xkcd_palette(["orange","yellow", "medium green","windows blue"])):
    '''
    Plot boxplots. Stack subplots horizontally.
    
    Arguments:
        data: List of dataframes. Length of data is the number of subplots, while each data array is inputs to each subplot.
        ylabels: List of strings. Y-labels of each subplots. Must be same length of $data$.
        xmin: Float or List. Min of x-axises.
        xmax: Float or List. Max of x-axises.
        toSaveFig: Bool. Whether to save the figure.
        title: String. If $toSaveFig$ is True, save the figure with filename of $title$
        palette_colors: palette colors of seaborn.

    
    '''
    #palette_colors2 = sns.color_palette(["#FF8000", "#FFFF00", "#00FF00", "#3333FF"])
    set_fig_fonts(20,24,26)
    # Create a figure instance, and the two subplots
    #xmin = 0.2
    #xmax = 1.025
    assert (len(data) ==len(ylabels)), "Error: len(data)!=len(names)"   
    n_subplot = len(ylabels)
    
    # handle xmin, xmax
    import numbers
    if isinstance(xmin, numbers.Number):
        xmin = [xmin] * len(ylabels)
    if isinstance(xmax, numbers.Number):
        xmax = [xmax] * len(ylabels)
    assert (len(xmin) ==len(ylabels)), "Error: len(xmin)!=len(names)" 
    assert (len(xmax) ==len(ylabels)), "Error: len(xmax)!=len(names)"  
    
    # start drawing
    fig = plt.figure(figsize=(12,15))
    
    for i in range(n_subplot):
        
        # add each subplot to fig        
        ax = fig.add_subplot(n_subplot,1,i+1)
        ax = sns.boxplot(data=data[i], orient="h", palette=palette_colors)#plot_kws={'line_kws':{'alpha':0.5}
        ax.xaxis.tick_top()
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(xmin[i],xmax[i])
        
        # if not the first subplot, switch off x-labelling
        if(i!=0):
            ax.tick_params(labelbottom='off')  
            
        plt.setp(ax.spines.values(), color='black')

    #plt.show()
    
    if(toSaveFig):
        title = title+'_boxplot'
        save_fig(fig,title)
    set_fig_fonts()

def load_and_clean_data(name, data_dir=None, encoding="ISO-8859-1", feature_col_num=1, result_offset=1, target="yield"):

    """
    Loads and cleans a CSV dataset for regression.
    
    Parameters:
        name (str): Base name of the CSV file (without extension).
        data_dir (str): Directory where the CSV file is stored. If None, uses the current working directory + '/data/'.
        encoding (str): File encoding.
        feature_col_num (int): Index of the first feature column.
        result_offset (int): The number of columns from the end to treat as non-target.
                             (target_col = total number of columns - result_offset)
        target (str): Which target column to use; must be either "yield" or "length".
    
    Returns:
        df_cleaned (pd.DataFrame): Cleaned DataFrame after duplicate removal.
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Regression target vector.
        clean_feature_list (list): List of feature column names.
        clean_result_col (str): The target column name used.
    """
    # Set directories and file path.
    root_dir = str(Path(os.getcwd()))
    if data_dir is None:
        data_dir = os.path.join(root_dir, "data")
    file_path = os.path.join(data_dir, f"{name}.csv")
    
    # Load the dataset.
    df = pd.read_csv(file_path, encoding=encoding)
    #print(f"Initial dataset size: {len(df)}")
    
    # Define indices for target columns.
    # yield_col_index is used for duplicate resolution.
    yield_col_index = len(df.columns) - result_offset - 1
    length_col_index = len(df.columns) - result_offset
    
    # Define feature columns (from feature_col_num up to the yield column).
    feature_list = df.columns[feature_col_num:yield_col_index]
    #print("Feature list:", list(feature_list))
    #print("Yield column (for duplicate resolution):", df.columns[yield_col_index])
    
    # Identify duplicates based on feature columns (ignoring the target column).
    exact_duplicates = df[df.duplicated(subset=list(feature_list), keep=False)]
    #print(exact_duplicates)
    
    # Check for conflicting target (yield) values among duplicates.
    duplicates = df.groupby(list(feature_list))[df.columns[yield_col_index]].nunique().reset_index()
    duplicates_with_diff = duplicates[duplicates[df.columns[yield_col_index]] > 1]
    #print(duplicates_with_diff)
    
    indices_to_keep = []
    indices_to_remove = []
    
    if not duplicates_with_diff.empty:
        duplicated_rows = df[df[list(feature_list)].apply(tuple, axis=1).isin(
            duplicates_with_diff[list(feature_list)].apply(tuple, axis=1)
        )]
        # Resolve conflicts: always compare using the yield column.
        for _, group in duplicated_rows.groupby(list(feature_list)):
            if 1 in group[df.columns[yield_col_index]].values:
                row_to_keep = group[group[df.columns[yield_col_index]] == 1].iloc[0]
                indices_to_keep.append(row_to_keep.name)
                rows_to_remove = group[group[df.columns[yield_col_index]] != 1].index
                indices_to_remove.extend(rows_to_remove)
            else:
                row_to_keep = group.iloc[0]
                indices_to_keep.append(row_to_keep.name)
                rows_to_remove = group.iloc[1:].index
                indices_to_remove.extend(rows_to_remove)
    
    # If no conflicting duplicates exist, keep only the first occurrence based on the feature set.
    indices_to_keep = df.drop_duplicates(subset=list(feature_list), keep="first").index
    indices_to_remove = df.index.difference(indices_to_keep)
    
    # Display the removed rows.
    removed_rows = df.loc[indices_to_remove]
    #print("\nNumber of Removed Duplicated Rows: ", len(removed_rows))
    #print(removed_rows)
    
    # Create the cleaned DataFrame.
    df_cleaned = df.drop(indices_to_remove).reset_index(drop=True)
    
    # Now select feature columns from the cleaned DataFrame.
    clean_feature_list = df_cleaned.columns[feature_col_num:yield_col_index]
    
    # Choose the target column after cleaning.
    if target == "yield":
        clean_result_col = df_cleaned.columns[yield_col_index]
    elif target == "length":
        clean_result_col = df_cleaned.columns[length_col_index]
    else:
        raise ValueError("target parameter must be either 'yield' or 'length'")
    
    X = df_cleaned[clean_feature_list].values
    Y = df_cleaned[clean_result_col].values
    
    non_zero_count = np.count_nonzero(Y)
    zero_count = (Y == 0).sum()
    
    # Print dataset summary.
    print(f"\nLoading {name} dataset...")
    print("Features list:", list(clean_feature_list))
    print("Feature size:", len(X[0]) if len(X) > 0 else 0)
    print("Dataset size:", len(Y))
    print(f"Target column: {clean_result_col}\nSum of target values: {np.sum(Y)}\nMean of target values: {np.mean(Y)}")
    print("Y non-zero:", non_zero_count, "\nY zero:", zero_count)
    
    return df_cleaned, X, Y, list(clean_feature_list), clean_result_col

def save_csv(data, title, ind=False):
    # Assume to_dir is defined in your global scope.
    to_save_title = format_title(to_dir, title, fileEtd='.csv')
    data.to_csv(to_save_title, index=ind)
    print('Successfully saved:', to_save_title)
    return to_save_title

def save_csv_subfolder_UF(data, *, title: str, UF: str, name: str,
                          ind: bool = False, out_dir: str = None) -> str:
    dir_eff = out_dir if out_dir is not None else to_dir
    path = format_title_subfolder_UF(dir_eff, title, UF=UF, name=name, file_ext=".csv")
    data.to_csv(path, index=ind)
    print("Successfully saved:", path)
    return path

from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

class MidpointNormalize(Normalize):
    """Normalise data so that *midpoint* → 0.5 in the colour map."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super(MidpointNormalize, self).__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        vmin, vmax, mid = self.vmin, self.vmax, self.midpoint

        # scale → 0‒1
        x = (result - vmin) / float(vmax - vmin)

        # split the range at the mid-point
        left  = (mid - vmin) / (vmax - vmin)
        mask  = x <= left
        x[mask]  *= 0.5 / left
        x[~mask] = 0.5 + (x[~mask] - left) * 0.5 / (1.0 - left)

        return np.ma.array(x, mask=result.mask, copy=False)

custom_div = LinearSegmentedColormap.from_list("purple_white_gold",["#f0bb3f", "#ffffff", "#4b1475"],N=256)

def latexize(name: str) -> str:
    if "_" not in name:
        return rf"$\mathrm{{{name}}}$"
    base, sub = name.split("_", 1)
    if sub.isnumeric():
        return rf"${base}_{{{sub}}}$"
    return rf"${base}_{{\mathrm{{{sub}}}}}$"

def plot_correlation_matrix(X, title, col_list, toSaveFig=True):
    df_std = pd.DataFrame(StandardScaler().fit_transform(X),columns=col_list)
    mat = df_std.corr().iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        mat,
        cmap=custom_div,
        norm=MidpointNormalize(vmin=-1, vmax=1, midpoint=0),
        interpolation="nearest"
    )

    latex_labels = [latexize(c) for c in col_list]
    latex_labels_rev = latex_labels[::-1]

    ax.set_xticks(np.arange(len(col_list)))
    ax.set_yticks(np.arange(len(col_list)))
    ax.set_xticklabels(latex_labels, rotation=-45, ha="left", fontsize=24)
    ax.set_yticklabels(latex_labels_rev, ha="right", fontsize=24)
    ax.grid(True, linewidth=0.5, alpha=0.2)

    cbar = fig.colorbar(im, boundaries=np.linspace(-1, 1, 21),
                        ticks=np.linspace(-1, 1, 5))
    #cbar.set_label("Pearson r", fontsize=14)
    cbar.ax.tick_params(labelsize=24)

    #plt.title("Pearson Correlation Matrix across Features", fontsize=24)
    if toSaveFig:
        save_fig(fig, f"{title}_corr_matrix")
    #plt.show()
