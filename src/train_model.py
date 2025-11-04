# src/train_model.py
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from mnist_loader import load_images, load_labels
from utils import ensure_dir, to_channels_last, save_model_keras

# Paths
TRAIN_IMAGES = "data/fashion/train-images-idx3-ubyte.gz"
TRAIN_LABELS = "data/fashion/train-labels-idx1-ubyte.gz"
TEST_IMAGES  = "data/fashion/t10k-images-idx3-ubyte.gz"
TEST_LABELS  = "data/fashion/t10k-labels-idx1-ubyte.gz"
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "fashion_mnist_cnn.keras")

# Ensure model dir exists
ensure_dir(MODEL_DIR)

# Load
X_train = load_images(TRAIN_IMAGES)
y_train = load_labels(TRAIN_LABELS)
X_test  = load_images(TEST_IMAGES)
y_test  = load_labels(TEST_LABELS)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# Preprocess
X_train = to_channels_last(X_train)  # (N,28,28,1), float32 [0,1]
X_test  = to_channels_last(X_test)

# Model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(
    X_train, y_train,
    epochs=5, batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# Save
save_model_keras(model, MODEL_PATH)

# Optional: quick evaluation print
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")