from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

def load_npy(filepath, label):
    def _load(fp, lbl):
        features = np.load(fp.numpy().decode('utf-8'))
        return features.astype(np.float32), lbl.numpy()

    features, lbls = tf.py_function(_load, [filepath, label], [tf.float32, tf.int32])
    features.set_shape([16, 2048])
    lbls.set_shape([])
    return features, lbls

def build_dataset(csv_path, batch_size=32, shuffle=True):
    
    df = pd.read_csv(csv_path)
    filepaths = df['filepath'].values
    labels = df['label'].values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.map(load_npy,  num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_model():
    model = Sequential([
        Input(shape=(16, 2048)),
        GRU(200),
        Dropout(0.5),
        Dense(101, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model

model = build_model()
print(model.summary())

df = pd.read_csv("/kaggle/working/Precomputed_Features/labels.csv")
train, val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'], shuffle=True)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)

train_dataset = build_dataset(csv_path="train.csv", batch_size=32, shuffle=True)
val_dataset = build_dataset(csv_path="val.csv", batch_size=32, shuffle=False)

history__ = model.fit(
    train_dataset,           
    epochs=20,    
    validation_data=val_dataset,
    
    callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
)

loss = history__.history["loss"]
acc = history__.history["accuracy"]
val_loss = history__.history["val_loss"]
val_acc = history__.history["val_accuracy"]

epochs = np.arange(1, len(loss) + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, marker='o', linestyle='-', linewidth=2,
         label="Train Loss")
plt.plot(epochs, val_loss, marker='s', linestyle='--', linewidth=2,
         label="Validation Loss")

plt.title("Training vs Validation Loss", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, acc, marker='o', linestyle='-', linewidth=2,
         label="Train Accuracy")
plt.plot(epochs, val_acc, marker='s', linestyle='--', linewidth=2,
         label="Validation Accuracy")

plt.title("Training vs Validation Accuracy", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()                    