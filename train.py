import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# -------------------------------
# Parameters
# -------------------------------
dataset_path = "dataset"
img_size = 224
batch_size = 32
epochs = 5

# -------------------------------
# Load Dataset
# -------------------------------
data = []
labels = []

print("📥 Loading images...")

# 0 = original, 1 = fake
for label, folder in enumerate(["original", "fake"]):
    folder_path = os.path.join('/home/rgukt/Documents/fake vs original screenshot detector/dataset', folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.resize(img, (img_size, img_size))
        data.append(img)
        labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print("Total images:", len(data))
print("Original:", np.sum(labels == 0))
print("Fake:", np.sum(labels == 1))

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, shuffle=True
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------------
# Build Model (Transfer Learning)
# -------------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# Train Model
# -------------------------------
print("🚀 Training model...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size
)

# -------------------------------
# Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# Save Model
# -------------------------------
model.save("model.h5")
print("💾 Model saved as model.h5")
