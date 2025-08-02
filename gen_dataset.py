import os
import cv2
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

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
    def __init__(self,
                 split='train',
                 skip=2,
                 mixup=True,
                 data_dir="../data",
                 pad_with_repeat=True):

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

    def create_features(self, feature):
        audio, label = feature

        audio = tf.cast(audio, tf.float32)
        audio = tf.numpy_function(butter_bandpass_filter, 
                                inp=[audio, lowcut, highcut, fs], 
                                Tout=tf.double)
        
        audio, _ = tf.numpy_function(librosa.effects.trim, 
                                    inp=[audio, 20], 
                                    Tout=[tf.double, tf.int64])
        
        if tf.shape(audio)[0] >= LENGHT:
            audio = audio[:LENGHT]
        else:
            diff = LENGHT - tf.shape(audio)[0]
            if self.pad_with_repeat:
                n_repetitions = tf.math.floordiv(LENGHT, tf.shape(audio)[0])
                if n_repetitions > 0:
                    audio = tf.tile(audio, [n_repetitions])
                audio = tf.pad(audio, paddings=[[0, LENGHT - tf.shape(audio)[0]]], mode='SYMMETRIC')
            else:
                audio = tf.pad(audio, paddings=[[0, diff]], mode='CONSTANT')
        
        if self.split == 'train':
            audio = self.augment_data(audio)
        
        image = tf.numpy_function(self.create_melspectrogram,
                                    inp=[audio],
                                    Tout=tf.float32)
        
        image = tf.cast(image, tf.float32) / 255.0

        image = tf.expand_dims(image, axis=-1)
        
        # label = 0 if subject is healthy, otherwise it's 1
        label = 0 if label == 0 else 1
        label = tf.one_hot(label, depth=2)

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
                # This is important to get (audio, label) tuples
                as_supervised=True
            )
            print(
                f"Loaded dataset split '{self.split}' with config '{config_name}'.")

        # The dataset should now yield tuples of (audio, label).
        # We can map over the dataset directly.
        print("Mapping dataset to create features...")
        # Ensure the dataset is not None before mapping

        data = self.dataset.map(lambda x, y: self.create_features(
            [x, y]), num_parallel_calls=tf.data.AUTOTUNE).shuffle(BATCH_SIZE * 20)

        # The rest of the code is unchanged
        data = data.shuffle(self.BATCH_SIZE * 10)
        if self.split == 'train':
            data = data.repeat()
        data = data.batch(self.BATCH_SIZE)
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
        return data
