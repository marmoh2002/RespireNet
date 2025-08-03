import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from gen_dataset import CoswaraCovidDataset
import tensorflow_datasets as tfds
# --- Configuration ---
# Directory to store the processed TFDS data

DATA_DIR = "/kaggle/working/tfds_dataset"

# Your local dataset path
LOCAL_DATASET_PATH = "/kaggle/working/dataset"

# --- 1. Instantiate the Dataset ---
# This will process your local audio files and create spectrograms
print("Initializing dataset...")
# Ensure the data_dir exists for the builder to write to
os.makedirs(DATA_DIR, exist_ok=True)
covid_dataset = CoswaraCovidDataset(split='train', data_dir=DATA_DIR)
dataset = covid_dataset.get_dataset()
print("Dataset initialized and processed.")


# --- 2. Visualize Spectrograms ---
# In your main script (buildd.py)

# --- 2. Visualize Spectrograms (Corrected Version) ---
print("\nVisualizing a batch of spectrograms and features...")

# Get one batch from the dataset
# The first element 'features' is now a dictionary.
for features, labels in dataset.take(1):
    # --- UNPACK THE DICTIONARY ---
    # Extract the image tensors and tabular data tensors from the dictionary
    image_batch = features['image_input']
    tabular_batch = features['tabular_input']

    # Now you can use .shape on the tensors
    print(f"Batch size: {image_batch.shape[0]}")
    print(f"Image Tensor Shape: {image_batch.shape}")
    print(f"Tabular Tensor Shape: {tabular_batch.shape}")
    print(f"Labels Tensor Shape: {labels.shape}")

    # --- VISUALIZE THE IMAGES ---
    plt.figure(figsize=(15, 12))  # Increased figure size for better layout
    for i in range(min(9, image_batch.shape[0])):  # Display up to 9 images
        ax = plt.subplot(3, 3, i + 1)

        # Get the specific image and tabular data for this example
        spectrogram_tensor = image_batch[i]
        tabular_features = tabular_batch[i]  # Shape: (4,)
        label_one_hot = labels[i]

        # Squeeze the channel dimension for plotting
        spectrogram = np.squeeze(spectrogram_tensor.numpy())
        plt.imshow(spectrogram.T, aspect='auto',
                   origin='lower', cmap='viridis')

        # --- DECODE AND DISPLAY ALL INFO IN THE TITLE ---
        # Decode the one-hot encoded label
        label_index = np.argmax(label_one_hot)
        label_name = 'COVID-19 Positive' if label_index == 1 else 'Healthy'

        # Decode the tabular features (age and sex)
        # Remember: age was normalized by 100, sex is one-hot encoded
        age = int(tabular_features[0].numpy() * 100)
        # Index of the '1' in the one-hot vector
        sex_index = np.argmax(tabular_features[1:].numpy())
        sex_map = {0: 'Male', 1: 'Female', 2: 'Other'}
        sex = sex_map[sex_index]

        # Create a detailed title
        plt.title(f"Label: {label_name}\nAge: {age}, Sex: {sex}")
        plt.xlabel("Time")
        plt.ylabel("Mels")
        plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()
