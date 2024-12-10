import numpy as np
import pandas as pd
from SourceFiles.utils import load_config

def clean_dataset(df, numeric_strategy='mean', outlier_method='zscore', z_thresh=3, verbose=True):
    """
    Cleans a dataset by handling missing values, ensuring correct data types, and removing outliers.

    Parameters:
    - df (pd.DataFrame): The dataset to clean.
    - numeric_strategy (str): Strategy for handling missing numeric values ('mean', 'median', or 'drop').
    - outlier_method (str): Method for detecting outliers ('zscore' or 'iqr').
    - z_thresh (float): Z-score threshold for outlier detection (used if method is 'zscore').
    - verbose (bool): If True, prints information about cleaning steps.

    Returns:
    - pd.DataFrame: The cleaned dataset.
    """
    
    # Handle missing values
    if verbose: print("Handling missing values...")
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                # Handle numeric missing values
                if numeric_strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif numeric_strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif numeric_strategy == 'drop':
                    df.dropna(subset=[col], inplace=True)
            else:
                # Handle categorical missing values
                df[col].fillna(df[col].mode()[0], inplace=True)
            if verbose: print(f" - Missing values in '{col}' handled.")

    # Ensure numeric columns are properly typed
    if verbose: print("\nEnsuring numeric columns are correctly typed...")
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            if verbose: print(f" - Ensured '{col}' is correctly typed")
        except Exception as e:
            if verbose: print(f" - Skipped '{col}': {e}")

    # Handle outliers
    if verbose: print("\nCleaning outliers...")
    if outlier_method == 'zscore':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > z_thresh
            df = df[~outliers]
            if verbose: print(f" - Removed outliers from '{col}' using Z-score (threshold={z_thresh}).")
    elif outlier_method == 'iqr':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            if verbose: print(f" - Removed outliers from '{col}' using IQR.")

    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)
    
    if verbose: print("\nData cleaning completed.\n")
    return df


def get_dataset():
    config = load_config()
    path = config["dataset_path"]
    df = pd.read_csv(path)
    clean_df  = clean_dataset (df)
    return clean_df



