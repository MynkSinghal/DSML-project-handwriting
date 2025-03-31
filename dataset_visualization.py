import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import pandas as pd

# Load MNIST dataset
print('Loading MNIST Dataset...')
dataset = fetch_openml('mnist_784')

# Create a figure with subplots
plt.figure(figsize=(15, 10))

# Plot 1: Sample digits
plt.subplot(2, 2, 1)
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(dataset.data.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f'Digit: {dataset.target[i]}')
    plt.axis('off')

# Plot 2: Digit distribution
plt.subplot(2, 2, 2)
digit_counts = dataset.target.value_counts().sort_index()
sns.barplot(x=digit_counts.index, y=digit_counts.values)
plt.title('Distribution of Digits')
plt.xlabel('Digit')
plt.ylabel('Count')

# Plot 3: Average digit images
plt.subplot(2, 2, 3)
for i in range(10):
    digit_data = dataset.data[dataset.target == str(i)]
    avg_digit = digit_data.mean().values.reshape(28, 28)
    plt.subplot(2, 5, i+1)
    plt.imshow(avg_digit, cmap='gray')
    plt.title(f'Avg {i}')
    plt.axis('off')

# Dataset statistics
stats = {
    'Total Samples': len(dataset.data),
    'Image Size': '28x28 pixels',
    'Number of Features': dataset.data.shape[1],
    'Number of Classes': len(dataset.target.unique()),
    'Class Distribution': dict(digit_counts)
}

# Create a table
plt.subplot(2, 2, 4)
plt.axis('off')
table_data = [[k, str(v)] for k, v in stats.items()]
table = plt.table(cellText=table_data,
                 colLabels=['Metric', 'Value'],
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.tight_layout()
plt.savefig('dataset_visualization.png')
plt.close()

# Save statistics to CSV
stats_df = pd.DataFrame(stats.items(), columns=['Metric', 'Value'])
stats_df.to_csv('dataset_statistics.csv', index=False) 