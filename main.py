# main.py

import os
import warnings
from src.data.make_dataset import load_data, preprocess_data, split_data, scale_features
from src.features.build_features import encode_features
from src.models.train_model import train_model
from src.models.predict_model import make_predictions
from src.visualization.visualize import plot_loss_curve, plot_scatter
from sklearn.metrics import mean_squared_error, r2_score

# Ignore all warnings
warnings.filterwarnings('ignore')

def main():
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    # Load and preprocess the data
    file_path = 'data/raw/Admission.csv'
    if os.path.exists(file_path):
        print(f"Found the file at: {file_path}")
    else:
        print(f"File not found at: {file_path}")
        
    data = load_data(file_path)
    data = preprocess_data(data)

    # Split the data into features and target
    xtrain, xtest, ytrain, ytest = split_data(data, 'Admit_Chance')
    
    # Encode features
    xtrain_encoded, encoder = encode_features(xtrain)
    xtest_encoded, _ = encode_features(xtest, encoder, fit=False)
    
    # Scale features
    Xtrain, Xtest = scale_features(xtrain_encoded, xtest_encoded)
    
    print("Training features shape:", Xtrain.shape)
    print("Test features shape:", Xtest.shape)
    
    # Perform Grid Search CV
    grid = train_model(Xtrain, ytrain)
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best Score: {-grid.best_score_}")  # Convert negative MSE to positive
    
    # Train the model with the best parameters
    best_model = grid.best_estimator_
    
    # Make predictions
    try:
        ypred = make_predictions(best_model, Xtest)
        mse = mean_squared_error(ytest, ypred)
        r2 = r2_score(ytest, ypred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
    except Exception as e:
        print("Error making predictions:", e)
    
    # Plot the loss curve
    plot_loss_curve(best_model)
    
    # Plot scatter plot
    plot_scatter(data)

if __name__ == "__main__":
    main()
