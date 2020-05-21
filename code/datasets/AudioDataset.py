import os, csv
import os.path
import json

from librosa.core import load
import torch.utils.data as data

from utils import text as t
from extract_durations import load_alignments


class AudioDataset(data.Dataset):
    """

    Expects the following structure

    root|- wavs
        |- metadata.csv
        |- durations.txt

    text.csv contains 3 columns - filename|text|tokenized-text (numbers to words etc.)
        The names must be alphabetically ordered
        The filename does not contain file type
        Columns are separated by pipe character

    wavs is a folder with audio files. Names correspond to `filename` field in text.csv

    durations.csv contain comma-separated integers, one row of integers per one row
        If use_phonemes is true, durations are expected to correspond to phonemes
        Else alignmnets are expected to correspond to text
    """

    def __init__(self, root, start_idx=0, end_idx=None, durations=False):
        super(AudioDataset, self).__init__()

        self.root = root
        self.start_idx, self.end_idx = start_idx, end_idx

        self.wavs = os.path.join(self.root, 'wavs')
        self.text = os.path.join(self.root, 'metadata.csv')
        self.durations = False if not durations else os.path.join(self.root, durations)

        self.wav_idx = sorted(os.listdir(self.wavs))[start_idx:end_idx]
        with open(self.text, encoding="utf-8") as txt:
            self.text_idx = [l.strip().split("|")[2].strip() for l in txt.readlines()][start_idx:end_idx]

        if self.durations:
            self.align_idx = load_alignments(self.durations)[start_idx:end_idx]

    def __getitem__(self, idx):
        wav = load(os.path.join(self.wavs, self.wav_idx[idx]))[0]
        text = self.text_idx[idx]
        alignment = self.align_idx[idx] if self.durations else None
        return text, wav, alignment

    def __len__(self):
        return len(self.wav_idx)
