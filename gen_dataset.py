import os
import cv2
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# from pydub import AudioSegment
# from pydub.silence import detect_leading_trailing_silence

# Import the custom dataset builder
import coswara

from filters import butter_bandpass_filter

# Assuming utils.filters is in a file named utils/filters.py
# You need to create this file if it doesn't exist.
# For now, I'll add a placeholder function.


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # This is a placeholder. You should use your actual implementation.
    # For demonstration, it just returns the data as is.
    return data


lowcut = 50.0
highcut = 8000.0
fs = 48000
LENGHT = 7 * fs
BATCH_SIZE = 16
PREFETCH_SIZE = 4
n_mels = 128
f_min = 50
f_max = 4000
nfft = 2048
hop = 512


class CoswaraCovidDataset:
    def __init__(self, split='train', skip=2, mixup=True, data_dir="../data", pad_with_repeat=True):

        self.split = split
        self.pad_with_repeat = pad_with_repeat
        self.data_dir = data_dir

        config_name = f'coughs-skip{skip}-{"mixup" if mixup else ""}'

        # Use the builder directly
        self.builder = coswara.Coswara(
            config=config_name, data_dir=self.data_dir)
        self.builder.download_and_prepare()

        self.dataset = self.builder.as_dataset(
            split=self.split, shuffle_files=True)

    def augment_data(self, audio):
        print("type of audio is: ", type(audio))
        audio = tf.cast(audio, tf.float32)
        p_roll = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        p_pitch_shift = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        if p_roll > 0.5:
            audio = tf.roll(audio, int(fs/10), axis=0)
        if p_pitch_shift > 0.3:
            # Ensure audio has a static shape for tf.numpy_function
            audio.set_shape([None])
            audio = tf.numpy_function(librosa.effects.pitch_shift,
                                      inp=[tf.cast(audio, tf.float32), fs, -2],
                                      Tout=tf.float32)
        return audio

    def create_melspectrogram(self, audio):
        # Ensure audio is a numpy array for librosa
        audio_np = audio if isinstance(audio, np.ndarray) else audio.numpy()
        image = librosa.feature.melspectrogram(y=audio_np.astype(np.float32),
                                               sr=fs, n_mels=n_mels,
                                               fmin=f_min, fmax=f_max,
                                               n_fft=nfft, hop_length=hop)
        image = librosa.power_to_db(image, ref=np.max)
        with np.errstate(divide='ignore', invalid='ignore'):
            image = np.nan_to_num(
                (image-image.min()) / (image.max() - image.min()), posinf=0.0, neginf=0.0)
        image *= 255
        return image.astype(np.float32)

    def create_features(self, audio, label):
        # Cast to float32 and apply bandpass filter

        audio = tf.cast(audio, tf.float32)
        audio = tf.numpy_function(
            butter_bandpass_filter,
            inp=[audio, lowcut, highcut, fs],
            Tout=tf.float32  # Changed from tf.double to tf.float32
        )

        # Trim audio - change output type to float32
        audio, _ = tf.numpy_function(
            librosa.effects.trim,
            inp=[audio, 20],
            Tout=[tf.float32, tf.int64]  # Changed from tf.double to tf.float32
        )

        # Standardize audio LENGHT
        if tf.shape(audio)[0] >= LENGHT:
            audio = audio[:LENGHT]
        else:
            diff = LENGHT - tf.shape(audio)[0]
            if self.pad_with_repeat:
                n_repetitions = tf.math.floordiv(LENGHT, tf.shape(audio)[0])
                if n_repetitions > 0:
                    audio = tf.tile(audio, [n_repetitions])
                audio = tf.pad(audio,
                               paddings=[[0, LENGHT - tf.shape(audio)[0]]],
                               mode='SYMMETRIC')
            else:
                audio = tf.pad(audio,
                               paddings=[[0, diff]],
                               mode='CONSTANT')

        # Apply augmentation for training
        if self.split == 'train':
            audio = self.augment_data(audio)

        # Create mel-spectrogram
        image = tf.numpy_function(
            self.create_melspectrogram,
            inp=[audio],
            Tout=tf.float32
        )

        # Normalize and format image
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=-1)

        # # Convert label to one-hot encoding
        # label = tf.cond(
        #     tf.equal(label, 0),
        #     lambda: tf.constant(0),
        #     lambda: tf.constant(1)
        # )
        # label = tf.one_hot(label, depth=2)
        label = tf.cond(
            tf.equal(label, 'healthy'),  # Direct string comparison
            lambda: tf.constant(0),
            lambda: tf.constant(1)
        )
        return image, label

    def get_dataset(self):
        """Builds and returns the tf.data.Dataset."""
        if self.dataset is None:
            config_name = f'coughs-skip{self.skip}-{"" if self.mixup else "no"}mixup'
            print(
                f"Loading dataset split '{self.split}' with config '{config_name}'...")
            self.dataset = tfds.load(
                'coswara',
                builder_kwargs={'config': config_name},
                split=self.split,
                data_dir=self.data_dir,
                shuffle_files=True,
            )
            print(
                f"Loaded dataset split '{self.split}' with config '{config_name}'.")

            # Debug: print dataset structure
            print("Dataset element spec:", self.dataset.element_spec)

        # Robust handling of dataset structure
        if 'audio' in self.dataset.element_spec and 'label' in self.dataset.element_spec:
            # Standard structure
            data = self.dataset.map(
                lambda x: self.create_features(x['audio'], x['label']),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        elif 'features' in self.dataset.element_spec and 'target' in self.dataset.element_spec:
            # Alternative structure
            data = self.dataset.map(
                lambda x: self.create_features(x['features'], x['target']),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # Fallback to first two elements
            data = self.dataset.map(
                lambda x: self.create_features(x[0], x[1]),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        data = data.shuffle(BATCH_SIZE * 10)
        if self.split == 'train':
            data = data.repeat()
        data = data.batch(BATCH_SIZE)
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
        return data
