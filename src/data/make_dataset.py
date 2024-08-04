import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess data: handle missing values, etc."""
    # Example preprocessing steps
    data = data.dropna()
    return data

def split_data(data, target_column):
    """Split data into features and target, then into train and test sets."""
    from sklearn.model_selection import train_test_split

    x = data.drop([target_column], axis=1)
    y = data[target_column]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)
    return xtrain, xtest, ytrain, ytest

def scale_features(xtrain, xtest):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(xtrain)
    Xtest = scaler.transform(xtest)
    return Xtrain, Xtest