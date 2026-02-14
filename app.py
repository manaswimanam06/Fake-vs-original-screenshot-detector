import tensorflow as tf
import cv2
import numpy as np
import sys

# Load trained model
model = tf.keras.models.load_model("model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found!")
        return
    
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))
    
    pred = model.predict(img)[0][0]

    if pred > 0.35:
        print("🛑 Prediction: FAKE Screenshot")
    else:
        print("✅ Prediction: ORIGINAL Screenshot")

# Run from command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py <image_path>")
    else:
        predict_image(sys.argv[1])
