import cv2
import numpy as np
import tensorflow as tf
import json
import sys
import os
import threading
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
from precompute_features import preprocess_video

model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/VideoCLF.keras"))
feature_extractor = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/resnet_50_feature_extractor.keras"))

with open(os.path.join(BASE_DIR, "label_map.json"), "r") as f:
    label_map = json.load(f)

reverse_label_map = {int(v): k for k, v in label_map.items()}

NUM_FRAMES = 16
CAPTURE_SECONDS = 3

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frames_to_capture = int(fps * CAPTURE_SECONDS)

frame_buffer = deque(maxlen=frames_to_capture)
predicted_class = "Waiting..."
confidence = 0.0
is_predicting = False  

def predict_worker():
    global predicted_class, confidence, is_predicting
    
    frames_snapshot = list(frame_buffer)
    
    indices = np.linspace(0, len(frames_snapshot) - 1, NUM_FRAMES).astype(int)
    sampled = [frames_snapshot[i] for i in indices]
    
    processed = preprocess_video(sampled)
    features = feature_extractor.predict(processed, verbose=0)
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features, verbose=0)
    predicted_label = np.argmax(predictions)
    
    predicted_class = reverse_label_map[int(predicted_label)]
    confidence = np.max(predictions) * 100
    is_predicting = False 

print("Press Q to quit")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append(frame.copy())
    frame_count += 1

    if frame_count % frames_to_capture == 0 and not is_predicting:
        if len(frame_buffer) == frames_to_capture:
            is_predicting = True
            thread = threading.Thread(target=predict_worker)
            thread.daemon = True  
            thread.start()

    indicator_color = (0, 0, 255) if is_predicting else (0, 255, 0)
    cv2.circle(frame, (20, 20), 10, indicator_color, -1)  
    cv2.rectangle(frame, (0, 35), (500, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Action: {predicted_class} ({confidence:.1f}%)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Video Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()