import cv2
import numpy as np
import time
import random
import pygame
from tensorflow.keras.models import load_model
import os

# === Paths ===
model_path = '/Users/atharvanaktode/Desktop/horse/horse_model.keras'
sounds_dir = '/Users/atharvanaktode/Desktop/horse/horse_noises'

# === Load model ===
model = load_model(model_path)

# === Load sounds ===
pygame.mixer.init()
sound_files = [
    os.path.join(sounds_dir, 'horse-neigh-2-390296.mp3'),
    os.path.join(sounds_dir, 'horse-neigh-shortened-84724.mp3'),
    os.path.join(sounds_dir, 'neighing-of-a-horse-154724.mp3')
]

# === Initialize camera ===
cap = cv2.VideoCapture(0)

# === Timing control for sound playback ===
last_neigh_time = 0
neigh_delay = 2  # seconds between neighs

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Preprocess frame for the model ===
    img = cv2.resize(frame, (128, 128))
    x = np.expand_dims(img / 255.0, axis=0)

    # === Predict ===
    pred_nohorse = model.predict(x, verbose=0)[0][0]
    pred_horse = 1 - pred_nohorse

    if pred_horse > 0.5:
        label = "horse"
        confidence = pred_horse
    else:
        label = "no_horse"
        confidence = pred_nohorse

    # === Play random neigh if confident horse ===
    current_time = time.time()
    if label == "horse" and confidence > 0.9:
        if current_time - last_neigh_time >= neigh_delay:
            sound_path = random.choice(sound_files)
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
            last_neigh_time = current_time

    # === Display results ===
    # === Big centered label ===
    text = f"{label.upper()} ({confidence*100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # bigger text
    thickness = 4      # bold
    color = (0, 255, 0) if label == "horse" else (0, 0, 255)

    # Get text size for centering
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    frame_height, frame_width, _ = frame.shape

    # Compute centered position
    text_x = (frame_width - text_width) // 2
    text_y = (frame_height + text_height) // 2

    # Draw text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow("🐴 Horse Detector", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
