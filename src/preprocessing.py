# preprocessing.py

import os
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# Define the augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

def augment_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)
    augmented = augment(samples=y, sample_rate=sr)
    sf.write(output_path, augmented, sr)
