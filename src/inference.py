from precompute_features import video_loader, preprocess_video
import tensorflow as tf
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/VideoCLF_v2.keras"))
feature_extractor = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/resnet_50_feature_extractor.keras"))

def predict_video(video_path, label_map):
    frames = video_loader(video_path)
    processed = preprocess_video(frames)  
    
    features = feature_extractor.predict(processed, verbose=0) 
    
    features = np.expand_dims(features, axis=0)  
    predictions = model.predict(features, verbose=0)  
    
    predicted_label = np.argmax(predictions)
    reverse_label_map = {v: k for k, v in label_map.items()}
    return reverse_label_map[predicted_label]

with open("label_map.json", "r") as f:
    label_map = json.load(f)

predicted_class = predict_video("example.avi", label_map)
print(f"Predicted: {predicted_class}")