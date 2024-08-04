from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

def train_model(Xtrain, ytrain):
    """Train the model with Grid Search CV."""
    param_grid = {
        'hidden_layer_sizes': [(3, 2), (5, 5), (7, 3)],
        'batch_size': [10, 20, 30],
        'max_iter': [100, 200, 300]
    }
    model = MLPRegressor()
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(Xtrain, ytrain)
    return grid_search