#!/usr/bin/env python3
"""
Example script to load MNIST data and visualize it.
"""

from data.mnist_loader import load_mnist
from utils.visualize import visualize_samples

if __name__ == "__main__":
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    
    print(f"Loaded {len(train_images)} training samples")
    print(f"Loaded {len(test_images)} test samples")
    
    # Visualize training samples
    print("Displaying training samples...")
    visualize_samples(train_images, train_labels, title="Training Samples", save_path="training_samples.png")
    
    # Visualize test samples
    print("Displaying test samples...")
    visualize_samples(test_images, test_labels, title="Test Samples", save_path="test_samples.png")
