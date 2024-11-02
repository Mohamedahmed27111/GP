import os
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define your mappings
original_mapping = {
    "1": "1", "6": "6", "5": "5", "4": "4", "7": "7", "9": "9", "3": "3", "2": "2", "8": "8",
    "meem": "م", "0": "0", "raa": "ر", "seen": "س", "alif": "ا", "daal": "د", "noon": "ن",
    "waw": "و", "ain": "ع", "haa": "ه", "laam": "ل", "jeem": "ج", "baa": "ب", "qaaf": "ق",
    "yaa": "ى", "faa": "ف", "Taa": "ط", "saad": "ص", "thaa": "ث", "ghayn": "ج", "sheen": "ش",
    "Thaa": "ظ", "khaa": "خ", "kaaf": "ك", "taa": "ت", "zay": "ز", "zaal": "ذ", "7aa": "ح",
    "daad": "ض"
}

# Arabic to English mapping
ar_to_en = {
    "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9", "٠": "0",
    "ا": "A", "ب": "B", "ت": "C", "ث": "D", "ج": "E", "ح": "F", "خ": "G", "د": "H", "ذ": "I",
    "ر": "J", "ز": "K", "س": "L", "ش": "M", "ص": "N", "ض": "O", "ط": "P", "ظ": "Q", "ع": "R", "غ": "S",
    "ف": "T", "ق": "U", "ك": "V", "ل": "W", "م": "n", "ن": "Y", "ه": "Z", "و": "a", "ى": "b",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "0": "0"
}

# Create a reverse mapping
en_to_ar = {v: k for k, v in ar_to_en.items()}

# Create a unique character vector
unique_letters = sorted(set(ar_to_en.values()))
unique_letters.append('X')  # Padding character
CHAR_VECTOR = ''.join(unique_letters)

# Load the CSV file with the training labels
train_labels_df = pd.read_csv('data/train/_annotations.csv')



# Update the label encoder to use the unique letters
label_encoder = LabelEncoder()
label_encoder.fit(unique_letters)  # Fit with unique letters

# Define a function to load images and their corresponding labels
def load_images_and_labels(image_dir, labels_df):
    images = []
    labels = []
    for index, row in labels_df.iterrows():
        image_path = os.path.join(image_dir, row['image_name'])
        image = cv2.imread(image_path)
        if image is not None:  # Check if the image was loaded successfully
            images.append(image)
            labels.append(row['label'])  # Assuming label is in a format that corresponds to the mapping
    return images, labels

# Load images and labels from the specified directory
image_dir = 'D:\college\GP 2\data\train'  # Adjust this path based on your directory structure
images, labels = load_images_and_labels(image_dir, train_labels_df)

# Preprocess images: resize and normalize
def preprocess_images(images):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (128, 64))  # Resize to 128x64 pixels
        normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
        resized_images.append(normalized_image)
    return np.array(resized_images)

# Preprocess the loaded images
processed_images = preprocess_images(images)

# Encode the labels as integers
encoded_labels = []
for label in labels:
    encoded_label = label_encoder.transform(list(label))  # Encode each character in the label
    encoded_labels.append(encoded_label)  # Append the encoded label

# Ensure all labels have the same length (padding if necessary)
max_text_len = 7
padded_labels = np.array([np.pad(enc_label, (0, max_text_len - len(enc_label)), 'constant') for enc_label in encoded_labels])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(processed_images, padded_labels, test_size=0.2, random_state=42)

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(unique_letters), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
