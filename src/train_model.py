# src/train_model.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from mnist_loader import load_images, load_labels
from utils import ensure_dir, to_channels_last, save_model_keras

# Paths
TRAIN_IMAGES = "data/fashion/train-images-idx3-ubyte.gz"
TRAIN_LABELS = "data/fashion/train-labels-idx1-ubyte.gz"
TEST_IMAGES  = "data/fashion/t10k-images-idx3-ubyte.gz"
TEST_LABELS  = "data/fashion/t10k-labels-idx1-ubyte.gz"
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "fashion_mnist_cnn.keras")
RESULTS_DIR  = "results"

# Ensure dirs exist
ensure_dir(MODEL_DIR)
ensure_dir(RESULTS_DIR)

# Load data
X_train = load_images(TRAIN_IMAGES)
y_train = load_labels(TRAIN_LABELS)
X_test  = load_images(TEST_IMAGES)
y_test  = load_labels(TEST_LABELS)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# Preprocess (to (N,28,28,1), float32 [0,1])
X_train = to_channels_last(X_train)
X_test  = to_channels_last(X_test)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# Model with dropout + batchnorm
model = keras.Sequential([
    layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train with generator
batch_size = 64
epochs = 10
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=1
)

# Save model
save_model_keras(model, MODEL_PATH)

# Predictions and visuals
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# Classification report
report = classification_report(y_test, y_pred, digits=4)
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# Accuracy/Loss curves
acc = history.history.get("accuracy")
val_acc = history.history.get("val_accuracy")
loss = history.history.get("loss")
val_loss = history.history.get("val_loss")

plt.figure()
plt.plot(acc, label="train_acc")
plt.plot(val_acc, label="val_acc")
plt.title("Accuracy")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"))
plt.close()

plt.figure()
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.title("Loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
plt.close()

# Final evaluation + metrics JSON
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

metrics = {
    "accuracy": float(test_acc),
    "loss": float(test_loss),
    "epochs": int(epochs),
    "train_samples": int(len(X_train)),
    "val_samples": int(len(X_test))
}
with open(os.path.join(RESULTS_DIR, "metrics_cnn.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)