import os
import cv2
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Import the custom dataset builder
import coswara

from filters import butter_bandpass_filter

# Placeholder filter function


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    return data

# --- MODIFIED PARAMETERS ---
# These parameters are adjusted to produce an image of shape (39, 88)
# Target shape: (height, width) -> (rows, cols) -> (39, 88)
# height -> n_mels
# width -> depends on LENGHT and hop size. width = floor(LENGHT / hop) + 1
# 88 = floor(LENGHT / 512) + 1 -> 87 = floor(LENGHT/512). Let LENGHT = 87 * 512 = 44544


lowcut = 50.0
highcut = 8000.0
fs = 48000
hop = 512
LENGHT = 44544  # MODIFIED: To get a spectrogram width of 88
BATCH_SIZE = 16
PREFETCH_SIZE = 4
n_mels = 39     # MODIFIED: To get a spectrogram height of 39
f_min = 50
f_max = 4000
nfft = 2048


class CoswaraCovidDataset:
    def __init__(self, split='train', skip=2, mixup=True, data_dir="../data", pad_with_repeat=True):
        self.split = split
        self.pad_with_repeat = pad_with_repeat
        self.data_dir = data_dir
        config_name = f'coughs-skip{skip}-{"mixup" if mixup else ""}'

        self.builder = coswara.Coswara(
            config=config_name, data_dir=self.data_dir)
        self.builder.download_and_prepare()
        self.dataset = self.builder.as_dataset(
            split=self.split, shuffle_files=True)

    def augment_data(self, audio):
        audio = tf.cast(audio, tf.float32)
        p_roll = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        p_pitch_shift = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        if p_roll > 0.5:
            audio = tf.roll(audio, int(fs/10), axis=0)
        if p_pitch_shift > 0.3:
            audio.set_shape([None])
            audio = tf.numpy_function(librosa.effects.pitch_shift,
                                      inp=[tf.cast(audio, tf.float32), fs, -2],
                                      Tout=tf.float32)
        return audio

    def create_melspectrogram(self, audio):
        audio_np = audio if isinstance(audio, np.ndarray) else audio.numpy()
        image = librosa.feature.melspectrogram(y=audio_np.astype(np.float32),
                                               sr=fs, n_mels=n_mels,
                                               fmin=f_min, fmax=f_max,
                                               n_fft=nfft, hop_length=hop)
        image = librosa.power_to_db(image, ref=np.max)

        # MODIFIED: Simplified normalization to [0, 1] range
        with np.errstate(divide='ignore', invalid='ignore'):
            image = (image - image.min()) / (image.max() - image.min())
        image = np.nan_to_num(image, posinf=0.0, neginf=0.0)

        return image.astype(np.float32)

    def create_features(self, audio, label):
        audio = tf.cast(audio, tf.float32)
        audio = tf.numpy_function(
            butter_bandpass_filter,
            inp=[audio, lowcut, highcut, fs],
            Tout=tf.float32
        )

        audio, _ = tf.numpy_function(
            librosa.effects.trim,
            inp=[audio, 20],
            Tout=[tf.float32, tf.int64]
        )

        # MODIFIED: Set a static shape for the audio tensor after trimming
        # This is important for subsequent processing steps.
        audio.set_shape([None])

        if tf.shape(audio)[0] >= LENGHT:
            audio = audio[:LENGHT]
        else:
            if self.pad_with_repeat:
                n_repetitions = tf.math.floordiv(LENGHT, tf.shape(audio)[0])
                if n_repetitions > 0:
                    audio = tf.tile(audio, [n_repetitions])
                audio = tf.pad(audio, paddings=[
                               [0, LENGHT - tf.shape(audio)[0]]], mode='SYMMETRIC')
            else:
                diff = LENGHT - tf.shape(audio)[0]
                audio = tf.pad(audio, paddings=[[0, diff]], mode='CONSTANT')

        # MODIFIED: Ensure final audio length is exactly LENGHT
        audio.set_shape([LENGHT])

        if self.split == 'train':
            audio = self.augment_data(audio)

        image = tf.numpy_function(
            self.create_melspectrogram,
            inp=[audio],
            Tout=tf.float32
        )

        # MODIFIED: Set the shape of the spectrogram explicitly
        # The shape is (height, width) or (n_mels, calculated_width)
        # calculated_width = 88 in our case
        image.set_shape([n_mels, 88])

        # MODIFIED: Convert 1-channel grayscale to 3-channel RGB format
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)

        # MODIFIED: Reshape to (rows, cols, channels) which is (39, 88, 3)
        image = tf.reshape(image, [n_mels, 88, 3])

        # MODIFIED: Change label to be a single integer (0 or 1) for binary_crossentropy
        label = tf.cond(
            tf.equal(label, 0),
            lambda: tf.constant(0, dtype=tf.float32),
            lambda: tf.constant(1, dtype=tf.float32)
        )

        return image, label

    def get_dataset(self):
        """Builds and returns the tf.data.Dataset."""
        data = self.dataset.map(
            lambda x: self.create_features(x['audio'], x['label']),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # Note: Batching is handled later in the as_numpy_arrays method
        # for compatibility with your KFold script.
        return data

    def as_numpy_arrays(self):
        """
        NEW METHOD: Converts the entire dataset into NumPy arrays.
        This is necessary for compatibility with sklearn's KFold.
        Warning: This will load the entire dataset into memory.
        """
        full_dataset = self.get_dataset()

        print("Converting tf.data.Dataset to NumPy arrays. This may take a moment...")

        images = []
        labels = []

        # Iterate over the dataset and collect all samples
        for image, label in full_dataset:
            images.append(image.numpy())
            labels.append(label.numpy())

        images_np = np.array(images)
        labels_np = np.array(labels)

        print(
            f"Conversion complete. Image shape: {images_np.shape}, Label shape: {labels_np.shape}")

        return images_np, labels_np
