import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
import pandas as pd

def create_hyperparameter_visualization():
    # Load MNIST dataset
    print('Loading MNIST Dataset...')
    dataset = fetch_openml('mnist_784')
    
    # Create a figure for hyperparameter visualization
    plt.figure(figsize=(15, 10))
    
    # KNN Hyperparameters
    plt.subplot(2, 2, 1)
    k_values = range(1, 11)
    accuracies = []
    
    # Simulate accuracy values (you should replace these with actual values)
    accuracies = [0.92, 0.94, 0.95, 0.96, 0.966, 0.965, 0.964, 0.963, 0.962, 0.961]
    
    plt.plot(k_values, accuracies, marker='o')
    plt.title('KNN: Accuracy vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    
    # CNN Hyperparameters
    plt.subplot(2, 2, 2)
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    
    # Create a heatmap of accuracies (simulated data)
    accuracies = np.array([
        [0.95, 0.96, 0.97],
        [0.96, 0.97, 0.98],
        [0.97, 0.98, 0.99]
    ])
    
    sns.heatmap(accuracies, annot=True, fmt='.2f', 
                xticklabels=batch_sizes, yticklabels=learning_rates)
    plt.title('CNN: Accuracy vs Learning Rate and Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    
    # Training Process Visualization
    plt.subplot(2, 2, 3)
    epochs = range(1, 21)
    train_loss = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06,
                  0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.018, 0.015, 0.012, 0.01]
    val_loss = [0.55, 0.45, 0.35, 0.3, 0.25, 0.2, 0.17, 0.15, 0.13, 0.11,
                0.1, 0.09, 0.085, 0.08, 0.075, 0.07, 0.068, 0.065, 0.062, 0.06]
    
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('CNN: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy over time
    plt.subplot(2, 2, 4)
    train_acc = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94,
                 0.95, 0.96, 0.965, 0.97, 0.975, 0.98, 0.982, 0.985, 0.988, 0.99]
    val_acc = [0.45, 0.55, 0.65, 0.7, 0.75, 0.8, 0.83, 0.85, 0.87, 0.89,
               0.9, 0.91, 0.915, 0.92, 0.925, 0.93, 0.932, 0.935, 0.938, 0.94]
    
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('CNN: Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_visualization.png')
    plt.close()

if __name__ == "__main__":
    create_hyperparameter_visualization() 