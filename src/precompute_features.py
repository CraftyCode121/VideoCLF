from tensorflow.keras.applications import ResNet50
from tqdm.notebook import tqdm
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import json
import os
import cv2

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

NUM_FRAMES = 16
SIZE = (224, 224)

def video_loader(path: str):
    cap = cv2.VideoCapture(str(path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-1, NUM_FRAMES).astype(int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])
    
    return frames[:NUM_FRAMES]

def preprocess_frame(frame):
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame =  cv2.resize(rgb_frame, SIZE)
    preprocessed = tf.keras.applications.resnet.preprocess_input(resized_frame)
    return preprocessed

def preprocess_video(frames):
    processed = [preprocess_frame(f) for f in frames]
    return np.stack(processed)

def build_feature_extractor():
    resnet_50 = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
    )
    resnet_50.trainable = False
    return resnet_50

def extract_features(processed_frames, verbose:int=0):
    features = feature_extractor.predict(processed_frames, verbose=verbose)
    return features

def load_and_preprocess(video_path):
    video_path = video_path.numpy().decode('utf-8')  
    frames = video_loader(video_path)
    processed = preprocess_video(frames)
    return processed

def precompute_features(dataset_path, save_path, batch_size=16):
    dataset_path = Path(dataset_path)
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    
    classes = sorted(os.listdir(dataset_path))
    label_map = {class_name: idx for idx, class_name in enumerate(classes)}
    
    all_paths = []
    all_labels = []
    all_save_files = []
    
    for classname in classes:
        class_folder = dataset_path / classname
        for video in class_folder.iterdir():
            all_paths.append(str(video))
            all_labels.append(label_map[classname])
            all_save_files.append(str(save_path / f"{classname}_{video.stem}.npy"))
    
    print(f"Total videos: {len(all_paths)}")
    
    dataset = tf.data.Dataset.from_tensor_slices(all_paths)
    dataset = dataset.map(
        lambda x: tf.py_function(load_and_preprocess, [x], tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE 
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) 
    
    records = []
    idx = 0
    for batch in tqdm(dataset, desc="Extracting features"):
        features_flat = feature_extractor.predict(batch.numpy().reshape(-1, 224, 224, 3), verbose=0)
        features_batch = features_flat.reshape(len(batch), 16, -1)
        
        for i in range(len(batch)):
            np.save(all_save_files[idx], features_batch[i])
            records.append((all_save_files[idx], all_labels[idx]))
            idx += 1
    
    df = pd.DataFrame(records, columns=["filepath", "label"])
    df.to_csv(save_path / "labels.csv", index=False)
    
    with open("/kaggle/working/label_map.json", "w") as f:
        json.dump(label_map, f)
        
    print("Done!")
    return label_map

#######################################################################
dataset_path = Path("ucf101/UCF101/UCF-101")
save_path = Path("Precomputed_Features")
label_map = precompute_features(dataset_path, save_path, batch_size=32)