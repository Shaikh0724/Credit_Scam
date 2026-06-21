import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_card_data(file_path):
    df = pd.read_csv(file_path)
    
    print("[DATASET PREVIEW]")
    print(df.head())
    print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n")
    
    print("[CLASS DISTRIBUTION DATA]")
    print(df['Class'].value_counts())
    print("\nExecuting preprocessing pipeline...")
    
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"Original Training Shape: {X_train.shape}")
    print(f"Balanced Resampled Training Shape: {X_train_res.shape}\n")
    
    X_train_seq = np.reshape(X_train_res, (X_train_res.shape[0], 1, X_train_res.shape[1]))
    X_test_seq = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    # Return statement ko simplified 6 variables par set kar diya hai
    return X_train_seq, X_test_seq, X_train_res, X_test, y_train_res, y_test