def make_predictions(model, Xtest):
    """Make predictions using the trained model."""
    ypred = model.predict(Xtest)
    return ypred