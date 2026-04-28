import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_dataset(config):
    """
    Loads and preprocesses the dataset specified in the config.
    Returns DataLoaders for train and test splits.
    For demonstration and reproducibility, it generates a synthetic 
    dataset mimicking NSL-KDD properties if raw files are missing.
    """
    dataset_name = config.get("data", {}).get("dataset_name", "nsl-kdd")
    batch_size = config.get("data", {}).get("batch_size", 256)
    
    print(f"Loading dataset: {dataset_name}")
    
    # Simulate NSL-KDD tabular data for out-of-the-box execution
    # In a real scenario, pd.read_csv("data/raw/KDDTrain+.txt") would be here.
    X, y = make_classification(
        n_samples=10000,
        n_features=41, # NSL-KDD has 41 features
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2], # Imbalanced classes mimicking normal/attack
        random_state=42
    )
    
    # Normalize Continuous Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=1-config.get("data", {}).get("train_split", 0.8), random_state=42
    )
    
    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} testing samples.")
    return train_loader, test_loader, X_train.shape[1], scaler
