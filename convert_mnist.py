import numpy as np
import struct
import pandas as pd

def read_idx(filename):
    """Read IDX file format"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def convert_mnist_to_csv():
    """Convert MNIST dataset to CSV files"""
    # Read training data
    print("Reading training images...")
    train_images = read_idx('dataset/train-images-idx3-ubyte')
    print("Reading training labels...")
    train_labels = read_idx('dataset/train-labels-idx1-ubyte')
    
    # Read testing data
    print("Reading test images...")
    test_images = read_idx('dataset/t10k-images-idx3-ubyte')
    print("Reading test labels...")
    test_labels = read_idx('dataset/t10k-labels-idx1-ubyte')
    
    # Reshape images to 2D array (samples x pixels)
    train_images_2d = train_images.reshape(train_images.shape[0], -1)
    test_images_2d = test_images.reshape(test_images.shape[0], -1)
    
    # Create column names for pixels
    pixel_columns = [f'pixel_{i}' for i in range(train_images_2d.shape[1])]
    
    # Create DataFrames
    print("Creating training CSV...")
    train_df = pd.DataFrame(train_images_2d, columns=pixel_columns)
    train_df.insert(0, 'label', train_labels)
    
    print("Creating test CSV...")
    test_df = pd.DataFrame(test_images_2d, columns=pixel_columns)
    test_df.insert(0, 'label', test_labels)
    
    # Save to CSV
    print("Saving training data to CSV...")
    train_df.to_csv('dataset/mnist_train.csv', index=False)
    print("Saving test data to CSV...")
    test_df.to_csv('dataset/mnist_test.csv', index=False)
    
    print("\nConversion completed!")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Create a small sample file for quick testing
    print("\nCreating sample files...")
    train_df.head(1000).to_csv('dataset/mnist_train_sample.csv', index=False)
    test_df.head(100).to_csv('dataset/mnist_test_sample.csv', index=False)
    print("Sample files created!")

if __name__ == "__main__":
    convert_mnist_to_csv()