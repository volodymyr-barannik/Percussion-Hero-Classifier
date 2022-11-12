import os

from pydub import AudioSegment
import math

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
