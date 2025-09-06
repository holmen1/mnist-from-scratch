# Script to download/load MNIST data

import os
import urllib.request
import gzip
import numpy as np
from pathlib import Path


def download_mnist():
    """Download MNIST dataset files if not present."""
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)

    for file in files:
        filepath = data_dir / file
        if not filepath.exists():
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
        else:
            print(f"{file} already exists.")


def load_images(filename):
    """Load MNIST images from idx3-ubyte file."""
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        # Read image data
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)

    return data.astype(np.float32) / 255.0  # Normalize to [0,1]


def load_labels(filename):
    """Load MNIST labels from idx1-ubyte file."""
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read label data
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)

    return data


def load_mnist():
    """Load MNIST dataset."""
    data_dir = Path(__file__).parent

    # Download if necessary
    download_mnist()

    # Load training data
    train_images = load_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(data_dir / "train-labels-idx1-ubyte.gz")

    # Load test data
    test_images = load_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    return (train_images, train_labels), (test_images, test_labels)


if __name__ == "__main__":
    # Quick test
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
