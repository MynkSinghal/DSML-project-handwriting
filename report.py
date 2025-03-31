import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_experimental_results_visualization():
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8')
    
    # Training percentages used in experiments
    training_percentages = [50, 60, 70, 80, 90]
    
    # Create figure for all metrics
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Sensitivity and Specificity
    plt.subplot(2, 2, 1)
    sensitivity = [0.022, 0.023, 0.075, 0.061, 0.068]
    specificity = [0.29, 0.31, 0.26, 0.29, 0.29]
    
    plt.plot(training_percentages, sensitivity, 'b-o', label='Sensitivity')
    plt.plot(training_percentages, specificity, 'r-o', label='Specificity')
    plt.title('Sensitivity and Specificity vs Training Percentage')
    plt.xlabel('Training Percentage')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Accuracy and F1-Score
    plt.subplot(2, 2, 2)
    accuracy = [0.27, 0.41, 0.39, 0.48, 0.45]
    f1_score = [0.302, 0.295, 0.272, 0.301, 0.301]
    
    plt.plot(training_percentages, accuracy, 'g-o', label='Accuracy')
    plt.plot(training_percentages, f1_score, 'm-o', label='F1-Score')
    plt.title('Accuracy and F1-Score vs Training Percentage')
    plt.xlabel('Training Percentage')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Error Rates (FDR, FNR)
    plt.subplot(2, 2, 3)
    fdr = [0.74, 0.727, 0.718, 0.689, 0.695]
    fnr = [0.71, 0.69, 0.74, 0.71, 0.71]
    
    plt.plot(training_percentages, fdr, 'c-o', label='FDR')
    plt.plot(training_percentages, fnr, 'y-o', label='FNR')
    plt.title('Error Rates vs Training Percentage')
    plt.xlabel('Training Percentage')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Additional Metrics (NPV, FPR)
    plt.subplot(2, 2, 4)
    npv = [0.022, 0.023, 0.075, 0.061, 0.068]
    fpr = [0.876, 0.875, 0.828, 0.839, 0.854]
    
    plt.plot(training_percentages, npv, 'k-o', label='NPV')
    plt.plot(training_percentages, fpr, 'orange', marker='o', label='FPR')
    plt.title('NPV and FPR vs Training Percentage')
    plt.xlabel('Training Percentage')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('report/experimental_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create comparison bar plot
    create_algorithm_comparison()

def create_algorithm_comparison():
    algorithms = ['KNN', 'SVM', 'Random Forest', 'CNN (TensorFlow)', 'CNN (Keras)']
    metrics = {
        'Accuracy': [96.67, 97.91, 96.82, 99.70, 98.75],
        'Precision': [96.5, 97.8, 96.7, 99.6, 98.6],
        'Recall': [96.7, 97.9, 96.8, 99.7, 98.7],
        'F1-Score': [96.6, 97.85, 96.75, 99.65, 98.65]
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algorithms))
    width = 0.15
    multiplier = 0
    
    for metric, scores in metrics.items():
        offset = width * multiplier
        ax.bar(x + offset, scores, width, label=metric)
        multiplier += 1
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison Across Different Algorithms')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(algorithms, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_experimental_results_visualization()