import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_curve(model):
    """Plot the loss curve of the model."""
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_scatter(data):
    """Plot a scatter plot of GRE_Score vs TOEFL_Score colored by Admit_Chance."""
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=data, 
                    x='GRE_Score', 
                    y='TOEFL_Score', 
                    hue='Admit_Chance')
    plt.title('GRE Score vs TOEFL Score by Admit Chance')
    plt.xlabel('GRE Score')
    plt.ylabel('TOEFL Score')
    plt.legend(title='Admit Chance')
    plt.grid(True)
    plt.show()