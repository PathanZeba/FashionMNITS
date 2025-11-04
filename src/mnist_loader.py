# src/mnist_loader.py
import gzip
import numpy as np

def load_images(path: str) -> np.ndarray:
    """Load IDX image file (.gz) and return shape (N, 28, 28)."""
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_labels(path: str) -> np.ndarray:
    """Load IDX label file (.gz) and return shape (N,)."""
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data