import os

from matplotlib import pyplot as plt
from pydub import AudioSegment
import math
import tensorflow_io as tfio
import tensorflow as tf

AudioSegment.ffmpeg = "E:\\Utility\\ffmpeg"

class SplitWavAudio():
    def __init__(self, source_folder, source_filename, target_folder, target_format="wav"):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.source_filename = source_filename
        self.source_filepath = os.path.join(source_folder, source_filename)
        print(f"Source audio is at {self.source_filepath}")

        self.source_audio = AudioSegment.from_file(self.source_filepath)
        self.target_format = target_format

    def get_duration(self):
        return self.source_audio.duration_seconds

    def single_split(self, from_sec, to_sec, target_filename, target_format):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.source_audio[t1:t2]

        os.makedirs(self.target_folder, exist_ok=True)
        split_audio.export(os.path.join(self.target_folder, target_filename), format=target_format)

    def multiple_split(self, sec_per_split):
        total_secs = math.ceil(self.get_duration())

        for i in range(0, total_secs, sec_per_split):
            target_filename = self.source_filename[0:-4] + "_" + str(i) + "." + self.target_format
            self.single_split(from_sec=i, to_sec=(i + sec_per_split), target_filename=target_filename, target_format=self.target_format)
            print(str(i) + 'th split is done')
            if i == (total_secs - sec_per_split):
                print('Audiofile was split successfully!')


source_folder = "Dataset"
sample_rate = 48000

def SplitAudio():
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            name, format = os.path.splitext(filename)
            if format == ".m4a":
                filepath = os.path.join(root, filename)
                target_dir = "Split" + root
                print(f"Processing \"{filename}\" at \"{root}\". Target directory is \"{target_dir}\"")

                split_wav = SplitWavAudio(source_folder=root, source_filename=filename, target_folder=target_dir)
                split_wav.multiple_split(sec_per_split=5)
            #print(os.path.join(filename, ""))


def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """

    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    # Scale magnitude relative to maximum value in S. Zeros in the output
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec


def GenerateSpectrograms():
    audio = tfio.audio.AudioIOTensor('SplitDataset/Cupped/P1_cupped_0.wav')
    audio = tfio.audio.AudioIOTensor('SplitDataset/Clapped/P1_clapped_0.wav')
    print(audio)

    audio = audio.to_tensor().numpy()
    audio = tf.squeeze(audio, axis=[-1])

    audio = tf.cast(audio, tf.float32) / 32768.0
    print(audio)
    audio.numpy()

    plt.figure()
    plt.plot(audio)
    spectrogram = tfio.audio.spectrogram(audio, nfft=512, window=512, stride=256)

    spectrogram = tf.signal.stft(audio, frame_length=1024, frame_step=512)

    magnitude_spectrograms = tf.abs(spectrogram)

    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(sample_rate=sample_rate,
                                                           lower_edge_hertz=40, upper_edge_hertz=21000,
                                                           num_spectrogram_bins=513, num_mel_bins=100)

    plt.figure()
    plt.imshow(mel_filterbank.numpy())
    plt.show()
    mel_power_spectrograms = tf.matmul(tf.square(magnitude_spectrograms), mel_filterbank)

    log_magnitude_mel_spectrograms = power_to_db(mel_power_spectrograms)
    spectrogram = log_magnitude_mel_spectrograms

    print(f"spectrogram={spectrogram.shape}")

    plt.figure()
    log_spectrogram = tf.math.log(spectrogram)
    print(f"log_spectrogram={log_spectrogram.shape}")
    show_spectrogram = tf.math.log(spectrogram, 10)
    show_spectrogram = tf.reverse(spectrogram, [1])
    show_spectrogram = tf.transpose(show_spectrogram)
    plt.imshow(show_spectrogram.numpy())

    plt.show()

GenerateSpectrograms()