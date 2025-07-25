import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

# Paths
dataset_path = os.path.join('.', 'data_cleaned', 'data_cleaned')

# Labels mapping
label_dict = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '-': 11, 'times': 12, 'lt': 13, 'gt': 14, 'neq': 15
}

# Data containers
print("[INFO] Loading and preprocessing images...")
data = []
labels = []

# Loop through folders
for label_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label_folder)

    if not os.path.isdir(folder_path):
        print(f"[WARNING] Skipping non-directory file: {label_folder}")
        continue

    if label_folder not in label_dict:
        print(f"[WARNING] Skipping unrecognized folder: {label_folder}")
        continue

    label_idx = label_dict[label_folder]

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"[WARNING] Unable to load image: {img_path}")
            continue

        kernel = np.ones((3, 3), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = 255 - image  # Invert
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)

        data.append(image)
        labels.append(label_idx)

# Final checks
if len(data) == 0:
    raise ValueError("[ERROR] No valid images found. Check dataset path and image files.")

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Model architecture
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(28, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("[INFO] Training model...")
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"[INFO] Test Accuracy: {acc:.4f} | Loss: {loss:.4f}")

# Save model
print("[INFO] Saving model...")
os.makedirs('../app', exist_ok=True)

# Save architecture
model_json = model.to_json()
with open("../app/model.json", "w") as json_file:
    json_file.write(model_json)

# Save weights
model.save_weights("model.weights.h5")
print("[INFO] Model saved successfully.")
