"""Extract durations for the LJSpeech dataset

usage: extract_durations.py [-h] [--durations_filename DURATIONS_FILENAME]
                            [--batch_size BATCH_SIZE]
                            checkpoint data_folder

positional arguments:
  checkpoint            Path to checkpoint of convolutional_cacotron model
  data_folder           Where the data live and where to save durations.

optional arguments:
  -h, --help            show this help message and exit
  --durations_filename DURATIONS_FILENAME
                        Name of the final durations file.
  --batch_size BATCH_SIZE
                        Batch size
"""

import os

import torch
import numpy as np
from barbar import Bar  # progress bar

def save_alignments_as_fertilities(model, dataloader, folder, durations_filename):
    """Save extracted alignments as durations
    
    Use the duration_Extraction model checkpoint to extract alignments and convert them into durations.
    For dataloader, use get_dataloader(64, 'cuda', start_idx=0, end_idx=13099, shuffle=False, sampler=SequentialSampler)
    """

    with open(os.path.join(folder, durations_filename), 'w') as file:
        for i, batch in enumerate(Bar(dataloader)):
            spectrs, slen, phonemes, plen, text = batch
            # supervised generation to get more reliable alignments
            out, alignment = model.generate(phonemes, plen, window=1, spectrograms=spectrs)
            fert = get_fertilities(alignment.cpu(), plen, slen)
            for f in fert:
                file.write(', '.join(str(x) for x in f) + '\n')


def get_fertilities(alignments, plen, slen):
    """Smoothed fertilities

    Values at indices correspond to fertilities for the phoneme at the given index.

    :param alignments: (batch, time, phoneme_len)
    :param plen: original phoneme length of each sentence in batch before padding
    :param slen: original spectrogram length before padding
    :return: list of 1D numpy arrays
    """
    fert = fertilities_improper(alignments, plen, slen)
    fert = smooth_fertilities(fert, slen)
    return fert

def fertilities_improper(alignments, plen, slen):
    """Phonemes not attended to get fertility one -> sum of fertilities may not equal slen

    Apply smoothing to get fertilities where sum of fertilities corresponds to number of spetrogram frames
    alignments must be non-decreasing! Enforce eg by monotonic attention

    :param alignments: (batch, time, phoneme_len)
    :return: fertilities: list of tensors
    """

    fertilities = []
    for i, a in enumerate(alignments):
        a = a[:slen[i], :plen[i]]
        # if frame is full of zeros, the attention went outside allowed range.
        # Att is monotonic -> place 1 to the end because the att will never come back -> focus on the last phoneme
        a[~(a>0).any(dim=1), -1] = 1
        am = torch.argmax(a, dim=-1)
        # expects sorted array
        uniq, counts = torch.unique_consecutive(am, return_counts=True)
        fert = torch.ones(plen[i], dtype=torch.long)  # bins for each phoneme
        fert[uniq] = counts
        fertilities.append(fert)

    return fertilities


def smooth_fertilities(fertilities_improper, slen):
    """Uniformly subtract 1 from n largest fertility bins, where n is the number of extra fertility points

    After smoothing, we should have sum(fertilities) = slen

    :param raw_fertilities: List of tensors from `fertilities_raw`
    :param slen: spectrogram lens
    :return: smooth_fertilities
    """

    smoothed = []
    for i, f in enumerate(fertilities_improper):
        ff = f.detach().cpu().numpy().copy()
        frames = slen[i]
        extra = ff.sum() - frames
        if extra:
            n_largest = np.argpartition(f, -extra)[-extra:]  # get `extra` largest fertilities indices
            ff[n_largest] -= 1
        smoothed.append(ff)

    return smoothed


def load_alignments(file):
    with open(file) as f:
        alignments = [[int(x) for x in l.split(',')] for l in f.readlines()]
    return alignments


def fert2align(fertilities):
    """Map list of fertilities to alignment matrix

    Allows backwards mapping for sanity check.

    :param fertilities: list of lists
    :return: alignment, list of numpy arrays, shape (batch, slen, plen)
    """

    alignments = []
    for f in fertilities:
        frames = np.sum(f.astype(int))
        a = np.zeros((frames, len(f)))
        x = np.arange(frames)
        y = np.repeat(np.arange(len(f)), f.astype(int))  # repeat each phoneme index according to fertiities
        a[(x, y)] = 1
        alignments.append(a)

    return alignments


def is_non_decreasing(x):
    """Check if values in x are non-decreasing

    :param x: 1D or 2D torch tensor, if 2D checked column-wise
    :return:
    """
    dx = x[1:] - x[:-1]
    return torch.all(dx >= 0, dim=0)


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.append('code')

    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SequentialSampler

    from datasets.AudioDataset import AudioDataset
    from duration_extractor import DurationExtractor, Collate

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint of convolutional_cacotron model")
    parser.add_argument("data_folder", type=str, help="Where the data live and where to save durations.")
    parser.add_argument("--durations_filename", default='durations.txt', type=str, help="Name of the final durations file.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    args = parser.parse_args()

    # Load pretrained checkpoint and extract alignments to data_folder
    m = DurationExtractor().load(args.checkpoint)
    dataset = AudioDataset(root=args.data_folder, durations=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=Collate(m.device),
                      shuffle=False, sampler=SequentialSampler(dataset))

    save_alignments_as_fertilities(m, dataloader, args.data_folder, args.durations_filename)
