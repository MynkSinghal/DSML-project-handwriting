import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
import networkx as nx

def create_knn_flow():
    G = nx.DiGraph()
    nodes = ['Input Image', 'Preprocessing', 'K-Nearest Neighbors', 'Distance Calculation', 'Majority Voting', 'Prediction']
    G.add_nodes_from(nodes)
    edges = [
        ('Input Image', 'Preprocessing'),
        ('Preprocessing', 'K-Nearest Neighbors'),
        ('K-Nearest Neighbors', 'Distance Calculation'),
        ('Distance Calculation', 'Majority Voting'),
        ('Majority Voting', 'Prediction')
    ]
    G.add_edges_from(edges)
    
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, arrowsize=20, font_size=10)
    plt.title('KNN Algorithm Flow')
    plt.savefig('knn_flow.png')
    plt.close()

def create_cnn_flow():
    G = nx.DiGraph()
    nodes = ['Input Image', 'Convolution Layer', 'ReLU', 'Pooling Layer', 
             'Flatten', 'Fully Connected', 'Output']
    G.add_nodes_from(nodes)
    edges = [
        ('Input Image', 'Convolution Layer'),
        ('Convolution Layer', 'ReLU'),
        ('ReLU', 'Pooling Layer'),
        ('Pooling Layer', 'Flatten'),
        ('Flatten', 'Fully Connected'),
        ('Fully Connected', 'Output')
    ]
    G.add_edges_from(edges)
    
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
            node_size=2000, arrowsize=20, font_size=10)
    plt.title('CNN Architecture Flow')
    plt.savefig('cnn_flow.png')
    plt.close()

def create_algorithm_descriptions():
    # Create a figure for algorithm descriptions
    plt.figure(figsize=(15, 10))
    
    # KNN Description
    plt.subplot(2, 2, 1)
    plt.axis('off')
    knn_text = """
    K-Nearest Neighbors (KNN)
    
    Algorithm:
    1. Load training data
    2. For each test instance:
       - Calculate distances to all training instances
       - Select K nearest neighbors
       - Majority vote for classification
    
    Key Equations:
    - Distance: d(x,y) = √Σ(xi-yi)²
    - Classification: Majority vote among K neighbors
    """
    plt.text(0.1, 0.9, knn_text, fontsize=10, verticalalignment='top')
    
    # CNN Description
    plt.subplot(2, 2, 2)
    plt.axis('off')
    cnn_text = """
    Convolutional Neural Network (CNN)
    
    Architecture:
    1. Input Layer (28x28)
    2. Convolution Layer
    3. ReLU Activation
    4. Pooling Layer
    5. Flatten
    6. Fully Connected Layer
    7. Output Layer (10 classes)
    
    Key Equations:
    - Convolution: (f * k)(p) = Σs f(s)k(p-s)
    - ReLU: f(x) = max(0,x)
    - Softmax: σ(z)j = e^zj / Σe^zk
    """
    plt.text(0.1, 0.9, cnn_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('algorithm_descriptions.png')
    plt.close()

if __name__ == "__main__":
    create_knn_flow()
    create_cnn_flow()
    create_algorithm_descriptions() 