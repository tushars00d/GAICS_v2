import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    Optimized for large-scale training using pin_memory and multiple workers.
    """
    dataset_name = config.get("data", {}).get("dataset_name", "nsl-kdd")
    batch_size = config.get("data", {}).get("batch_size", 1024)
    num_workers = config.get("data", {}).get("num_workers", 2)
    pin_memory = config.get("data", {}).get("pin_memory", True)
    mock_samples = config.get("data", {}).get("mock_samples", 100000)
    
    print(f"Loading dataset: {dataset_name} (Simulating {mock_samples} samples)")
    
    # Scale up synthetic generation to stress test pipelines
    X, y = make_classification(
        n_samples=mock_samples,
        n_features=41,
        n_informative=20,
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
    
    # Optimized DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} testing samples.")
    return train_loader, test_loader, X_train.shape[1], scaler
