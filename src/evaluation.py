import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .train import build_dataset
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/VideoCLF_v2.keras"))

val_dataset = build_dataset(csv_path="val.csv", batch_size=32, shuffle=False)

with open("label_map.json", "r") as f:
    label_map = json.load(f)

if isinstance(list(label_map.values())[0], int):
    index_to_class = {v: k for k, v in label_map.items()}
else:
    index_to_class = {int(k): v for k, v in label_map.items()}
    label_map = {v: k for k, v in index_to_class.items()}

num_classes = len(index_to_class)

y_pred_probs = model.predict(val_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = np.concatenate([y for x, y in val_dataset], axis=0)

if len(y_true.shape) > 1:
    y_true = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(10, 10))
plt.imshow(cm_normalized)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.tight_layout()
plt.show()

class_names = [index_to_class[i] for i in range(num_classes)]

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

total_per_class = cm.sum(axis=1)
correct_per_class = np.diag(cm)

error_rate = 1 - (correct_per_class / total_per_class)

error_series = pd.Series(error_rate, index=class_names)

top_10_worst = error_series.sort_values(ascending=False).head(10)

print("\nTop 10 Most Confused Classes (by error rate):\n")
print(top_10_worst)

plt.figure(figsize=(12, 6))
plt.bar(top_10_worst.index, top_10_worst.values)
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Most Confused Classes (Error Rate)")
plt.ylabel("Error Rate")
plt.grid(True)
plt.tight_layout()
plt.show()

worst_class_name = top_10_worst.index[0]
worst_class_index = label_map[worst_class_name]

confusions = cm[worst_class_index].copy()
confusions[worst_class_index] = 0  

confusion_series = pd.Series(confusions, index=class_names)

top_confusions = confusion_series.sort_values(ascending=False).head(5)

print(f"\nClass '{worst_class_name}' is most confused with:\n")
print(top_confusions)