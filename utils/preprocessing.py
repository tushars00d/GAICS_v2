import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def normalize_continuous(df, continuous_columns):
    """
    Helper for Layer 1 to normalize continuous tabular features.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[continuous_columns] = scaler.fit_transform(df[continuous_columns])
    return df, scaler

def encode_categorical(df, categorical_columns):
    """
    Encodes categorical variables into integer representations.
    """
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders
