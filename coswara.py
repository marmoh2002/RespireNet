import random
import pathlib
import logging
import tensorflow_datasets as tfds
from pydub import AudioSegment
from pydub.utils import make_chunks
import tensorflow as tf
# ... (Keep the _DESCRIPTION, _CITATION, LABEL_MAP, etc. the same as before)
_DESCRIPTION = """
Drawn by the information packed nature of sound signals, Project Coswara aims to evaluate
effectiveness for COVID-19 diagnosis using sound samples. The idea is to create a huge dataset
of breath, cough, and speech sound, drawn from healthy and COVID-19 positive individuals,
from around the globe. Subsequently, the dataset will be analysed using signal processing
and machine learning techniques for evaluating the effectiveness in automatic detection of
respiratory ailments, including COVID-19.
"""

_CITATION = """
@article{Sharma_2020,
   title={Coswara — A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis},
   url={http://dx.doi.org/10.21437/Interspeech.2020-2768},
   DOI={10.21437/interspeech.2020-2768},
   journal={Interspeech 2020},
   publisher={ISCA},
   author={Sharma, Neeraj and Krishnan, Prashant and Kumar, Rohit and Ramoji, Shreyas and Chetupalli, Srikanth Raj and R., Nirmala and Ghosh, Prasanta Kumar and Ganapathy, Sriram},
   year={2020},
   month={Oct}
}
"""

LABEL_MAP = {
    'healthy': 0,
    'positive_moderate': 1,
    'positive_mild': 2,
    'positive_asymp': 3
}

SAMPLE_RATE = 48000
POSSIBLE_SKIPS = range(1, 6)
CHUNK_LENGHT_MS = 3000


class CoswaraConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Coswara Dataset."""

    def __init__(self, name, skip, mixup=False, **kwargs):
        """Constructs a CoswaraConfig."""
        mixup_string = 'mixup' if mixup else ''
        super(CoswaraConfig, self).__init__(
            name=f"{name}-skip{skip}-{mixup_string}",
            version=tfds.core.Version("1.0.0"),
            description=f'Coswara dataset for {name}.',
            **kwargs,
        )
        self.skip = skip
        self.mixup = mixup


class Coswara(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Coswara dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release with support for local cough datasets.',
    }

    BUILDER_CONFIGS = [
        CoswaraConfig(name='coughs', skip=i, mixup=m)
        for i in POSSIBLE_SKIPS
        for m in [True, False]
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'label': tfds.features.ClassLabel(names=['healthy', 'positive_moderate', 'positive_mild', 'positive_asymp']),
                'audio': tfds.features.Audio(file_format='wav', sample_rate=SAMPLE_RATE),
                'user_id': tfds.features.Text(),
                'age': tf.int64,
                'sex': tfds.features.ClassLabel(names=['male', 'female']),
            }),
            supervised_keys=None,  # ('audio', 'label'),
            homepage='https://coswara.iisc.ac.in/?locale=en-US',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        base_path = pathlib.Path('/kaggle/working/dataset')
        return {
            'train': self._generate_examples(base_path / 'train'),
            'test': self._generate_examples(base_path / 'test'),
            'validation': self._generate_examples(base_path / 'validation'),
        }

    def _generate_examples(self, path):
        """Yields examples and logs its progress."""
        log_file = '/kaggle/working/generator.log'
        # Clear log file for the first run (train split)
        if path.name == 'train' and pathlib.Path(log_file).exists():
            open(log_file, 'w').close()

        logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"--- Generator starting for split: {path.name} ---")
        logging.info(f"Looking in path: {path}")

        if not path.exists():
            logging.error(f"Path does not exist: {path}")
            return

        # For simplicity in debugging, we'll temporarily disable mixup logic
        file_list = list(path.iterdir())
        logging.info(f"Found {len(file_list)} items in the directory.")

        total_yielded = 0
        for user_dir in file_list:
            logging.info(f"Processing item: {user_dir.name}")
            if not user_dir.is_dir():
                logging.warning(f"  -> Item is not a directory, skipping.")
                continue

            label_file = user_dir / 'label.txt'
            wav_files = list(user_dir.glob('*.wav'))
            metadata_file = user_dir / 'metadata.csv'

            if not label_file.exists():
                logging.warning(
                    f"  -> FAIL: 'label.txt' not found in {user_dir.name}.")
                continue
            if not all([label_file.exists(), wav_files, metadata_file.exists()]):
                logging.warning(
                    f"Skipping {user_dir.name} due to missing files.")
                continue

            audio_file = wav_files[0]
            try:
                metadata = pd.read_csv(metadata_file).iloc[0]
                age = int(metadata['age'])
                # Ensure lowercase for consistency
                sex = metadata['sex'].lower()

                duration = AudioSegment.from_file(audio_file).duration_seconds
                if duration >= self.builder_config.skip:
                    logging.info(
                        f"  -> SUCCESS: Yielding {user_dir.name} (duration: {duration:.2f}s)")
                    total_yielded += 1
                    yield user_dir.name, {
                        'label': LABEL_MAP[open(label_file).read().strip()],
                        'audio': audio_file,
                        'user_id': user_dir.name,
                        'age': age,
                        'sex': sex,
                    }
                else:
                    logging.warning(
                        f"  -> FAIL: Duration for {user_dir.name} is {duration:.2f}s, required {self.builder_config.skip}s.")
            except Exception as e:
                logging.error(
                    f"  -> CRITICAL FAIL: Could not process audio file {audio_file}. Error: {e}")

        logging.info(
            f"--- Generator finished for split: {path.name}. Total yielded: {total_yielded} ---")

    # Keep the rest of the file (_get_nonempty_chunk_list, _generate_mixup_examples) as it was
    def _get_nonempty_chunk_list(self, users):
        chunks = []
        user = None
        while len(chunks) == 0:
            user = random.choice(users)
            wav_files = list(user.glob('*.wav'))
            if not wav_files:
                users.remove(user)
                if not users:
                    return None, []
                continue
            audio = AudioSegment.from_file(wav_files[0], "wav")
            chunks = make_chunks(audio, CHUNK_LENGHT_MS)
        return user, chunks

    def _generate_mixup_examples(self, path):
        save_generated = path.parent / 'new'
        # This function is not called in the logging version to simplify debugging
        pass
