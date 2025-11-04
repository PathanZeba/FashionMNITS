 # src/utils.py
import os

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def to_channels_last(x):
    """Ensure input is float32 [0,1] and has channels-last shape for CNN."""
    import numpy as np
    x = x.astype("float32") / 255.0
    return np.expand_dims(x, -1)  # (N,28,28) -> (N,28,28,1)

def save_model_keras(model, path: str) -> None:
    """Save in modern Keras format (.keras)."""
    ensure_dir(os.path.dirname(path))
    model.save(path)
    print(f"Model saved at {path}")
