import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_features(data, encoder=None, fit=True):
    """Encode categorical features using OneHotEncoder."""
    if fit:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(data)
    else:
        encoded_data = encoder.transform(data)
    
    return pd.DataFrame(encoded_data), encoder
