import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from coswara_dataset import CoswaraCovidDataset

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
print("\nVisualizing a batch of spectrograms...")

# Get one batch from the dataset
for images, labels in dataset.take(1):
    plt.figure(figsize=(15, 10))
    for i in range(min(9, images.shape[0])):  # Display up to 9 images
        ax = plt.subplot(3, 3, i + 1)
        # Squeeze the channel dimension for plotting
        spectrogram = np.squeeze(images[i].numpy())
        plt.imshow(spectrogram.T, aspect='auto',
                   origin='lower', cmap='viridis')

        # Decode the one-hot encoded label
        label_index = np.argmax(labels[i])
        label_name = 'COVID-19 Positive' if label_index == 1 else 'Healthy'
        plt.title(f"Label: {label_name}")
        plt.xlabel("Time")
        plt.ylabel("Mels")
        plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()
