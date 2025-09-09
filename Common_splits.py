import random
import pickle
import os

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import utils

name = "CrTeNW_data"

root_dir = str(Path(os.getcwd()))
from_dir = root_dir + "/data/"
to_dir = root_dir + "/results/"

def generate_alternating_kmeans_splits(num_splits, X, init_train_size, exclude_index=None, seed=42):
    """
    Generate a list of (train_indices, test_indices) pairs using an alternating
    selection method based on k-means clustering, with an option to exclude a designated index.
    
    Steps:
      1. Run k-means clustering on X to obtain clusters (number of clusters = init_train_size).
      2. For each cluster, shuffle the indices of data points in that cluster.
      3. For each split, select one index from each cluster using round-robin selection.
         If the candidate equals exclude_index, skip it and choose the next one.
         Wrap around if needed.
      4. The union of these selected indices forms the training set;
         all remaining indices become the test set.
    
    Parameters:
        num_splits (int): Number of splits to generate.
        X (array-like): Feature matrix of shape (n_samples, n_features).
        init_train_size (int): Number of initial training points (also the number of clusters).
        exclude_index (int, optional): An index to be excluded from the training set.
        seed (int): Base random seed.
        
    Returns:
        splits (list): A list of tuples, where each tuple is (train_indices, test_indices).
    """
    import random
    random.seed(seed)
    n_samples = X.shape[0]
    all_indices = list(range(n_samples))
    
    # Step 1: Run k-means clustering to form 'init_train_size' clusters.
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=init_train_size, random_state=seed)
    clusters = kmeans.fit_predict(X)
    
    # Organize indices by cluster.
    cluster_dict = {i: [] for i in range(init_train_size)}
    for idx, cl in enumerate(clusters):
        cluster_dict[cl].append(idx)
    
    # Step 2: Shuffle indices in each cluster.
    for cl in cluster_dict:
        random.shuffle(cluster_dict[cl])
    
    # Prepare counters for round-robin selection.
    cluster_counters = {i: 0 for i in range(init_train_size)}
    
    splits = []
    for split in range(num_splits):
        train_indices = []
        for cl in range(init_train_size):
            indices = cluster_dict[cl]
            candidate = None
            counter = cluster_counters[cl]
            # Try to find a candidate that is not the exclude_index.
            for offset in range(len(indices)):
                idx_candidate = indices[(counter + offset) % len(indices)]
                if idx_candidate != exclude_index:
                    candidate = idx_candidate
                    cluster_counters[cl] = counter + offset + 1  # Update counter for this cluster.
                    break
            # If every candidate in this cluster equals exclude_index (very unlikely),
            # fall back to a candidate.
            if candidate is None:
                candidate = indices[cluster_counters[cl] % len(indices)]
                cluster_counters[cl] += 1
            train_indices.append(candidate)
        # Ensure uniqueness (should be unique, one per cluster).
        train_indices = list(set(train_indices))
        test_indices = [x for x in all_indices if x not in train_indices]
        splits.append((train_indices, test_indices))
    
    return splits


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
    print(f"Initial dataset size: {len(df)}")
    
    # Define indices for target columns.
    # yield_col_index is used for duplicate resolution.
    yield_col_index = len(df.columns) - result_offset - 1
    length_col_index = len(df.columns) - result_offset
    
    # Define feature columns (from feature_col_num up to the yield column).
    feature_list = df.columns[feature_col_num:yield_col_index]
    print("Feature list:", list(feature_list))
    print("Yield column (for duplicate resolution):", df.columns[yield_col_index])
    
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
    print("\nNumber of Removed Duplicated Rows: ", len(removed_rows))
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

def plot_all_splits_with_clusters_PCA(X, splits, init_train_size, output_filepath, ncols=3, to_save=False):
    """
    Visualize all generated splits along with k-means clustering.
    
    Parameters:
        X (array-like): The full feature matrix (n_samples, n_features).
        splits (list): A list of (train_indices, test_indices) pairs.
        init_train_size (int): Number of clusters/initial training points.
        ncols (int): Number of columns for subplots.
    """
    # Run k-means on the entire dataset to get cluster labels and centers.
    kmeans = KMeans(n_clusters=init_train_size, random_state=42)
    clusters = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # Use PCA to reduce to 2D.
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    centers_reduced = pca.transform(centers)
    
    num_splits = len(splits)
    nrows = int(np.ceil(num_splits / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    
    for idx, (train_indices, test_indices) in enumerate(splits):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]
        
        # Plot all data points colored by cluster.
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', alpha=0.3)
        # Plot cluster centers.
        ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='red', marker='X', s=150, label='Cluster Centers')
        # Highlight the selected training points for this split.
        selected_points = X_reduced[train_indices]
        ax.scatter(selected_points[:, 0], selected_points[:, 1], color='black', s=80, label='Selected Points')
        
        ax.set_title(f"Split {idx+1}")
        ax.legend()
    
    # Remove any unused subplot axes.
    for idx in range(num_splits, nrows*ncols):
        fig.delaxes(axes.flatten()[idx])
    
    plt.tight_layout()
    plt.colorbar(scatter, ax=axes, fraction=0.02, pad=0.04, label="Cluster Label")
    
    
    if to_save:
        plt.savefig(output_filepath, dpi=300)
        print("Figure saved to:", output_filepath)

    #plt.show()

def plot_all_splits_with_clusters_tsne(X, splits, init_train_size, output_filepath, ncols=3, to_save=False):
    """
    Visualize all generated splits along with k-means clustering using t-SNE for dimensionality reduction.
    The figure is saved to output_filepath if to_save is True.
    
    Parameters:
        X (array-like): The full feature matrix (n_samples, n_features).
        splits (list): A list of (train_indices, test_indices) pairs.
        init_train_size (int): Number of clusters/initial training points.
        output_filepath (str): File path to save the figure.
        ncols (int): Number of columns in the subplot grid.
        to_save (bool): If True, save the figure.
    """
    # Run k-means on the entire dataset.
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=init_train_size, random_state=42)
    clusters = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # Compute t-SNE embedding for the full dataset.
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X)
    
    # For each cluster, find the point closest to the cluster center in the original space.
    representative_indices = []
    for cl in range(init_train_size):
        cluster_indices = np.where(clusters == cl)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_points = X[cluster_indices]
        distances = np.linalg.norm(cluster_points - centers[cl], axis=1)
        rep_idx = cluster_indices[np.argmin(distances)]
        representative_indices.append(rep_idx)
    
    centers_reduced = X_reduced[representative_indices]
    
    num_splits = len(splits)
    nrows = int(np.ceil(num_splits / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    
    # Use a colormap to assign a unique color for each split.
    cmap = get_cmap("tab10")
    
    for idx, (train_indices, test_indices) in enumerate(splits):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]
        
        # Plot all data points colored by cluster.
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', alpha=0.3)
        # Plot cluster centers.
        ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='red', marker='X', s=150, label='Cluster Centers')
        # Highlight the selected training points for this split.
        color = cmap(idx % 10)
        selected_points = X_reduced[train_indices]
        ax.scatter(selected_points[:, 0], selected_points[:, 1], color=color, s=80, edgecolor='k', label=f"Split {idx+1} Selected")
        ax.set_title(f"Split {idx+1}")
        ax.legend()
    
    # Remove unused subplots.
    for idx in range(num_splits, nrows*ncols):
        fig.delaxes(axes.flatten()[idx])
    
    plt.tight_layout()
    plt.colorbar(scatter, ax=axes, fraction=0.02, pad=0.04, label="Cluster Label")
    
    if to_save:
        plt.savefig(output_filepath, dpi=300)
        print("Figure saved to:", output_filepath)
    
    #plt.show()

df_cleaned, X, Y, clean_feature_list, clean_result_col = utils.load_and_clean_data(name, target="length")

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X = X_normalized

inner_nsplits = 10
init_train_size = 20
totalSamp = X.shape[0]

Y_global_max = np.max(Y)
idx_global_max = np.argmax(Y)
all_ind = np.random.permutation(list(range(0,totalSamp)))
all_ind_wo_max = list(range(0,totalSamp))
all_ind_wo_max.remove(0)

# Generate Common Split with kmeans-clustering - avoiding concentrated sampling
num_splits = 10
common_splits = generate_alternating_kmeans_splits(num_splits, X, init_train_size, exclude_index=idx_global_max, seed=42)

# Save the splits into pkl files
splits_filepath = os.path.join(from_dir, "common_splits_1.pkl")
with open(splits_filepath, 'wb') as f:
    pickle.dump(common_splits, f)
print("Initial splits saved to:", splits_filepath)

tsne_output_filepath = os.path.join(to_dir, "diverse_initial_splits_tsne.png")
pca_output_filepath = os.path.join(to_dir, "diverse_initial_splits_PCA.png")

plot_all_splits_with_clusters_tsne(X, common_splits, init_train_size, tsne_output_filepath, ncols=3, to_save=True)
plot_all_splits_with_clusters_PCA(X, common_splits, init_train_size, pca_output_filepath, ncols=3, to_save=True)