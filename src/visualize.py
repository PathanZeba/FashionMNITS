# src/visualize.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_training(history, out_png="results/training_curves.png"):
    ensure_dir(os.path.dirname(out_png))
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(acc, label="train acc")
    plt.plot(val_acc, label="val acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved training curves: {out_png}")

def plot_confusion(y_true, y_pred, out_png="results/confusion_matrix.png"):
    ensure_dir(os.path.dirname(out_png))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved confusion matrix: {out_png}")