import torch
from random import sample, randrange


def add_random_noise(spectrograms, std_dev):
    """Add noise from Normal(0, std_dev)

    :param spectrograms:
    :param std_dev:
    :return:
    """
    if not std_dev: return spectrograms
    return spectrograms + std_dev * torch.randn(spectrograms.shape).to(spectrograms.device)


def degrade_some(model, spectrograms, phonemes, plen, ratio, repeat=1):
    """Replace some spectrograms in batch by their generated equivalent

    Ideally, run this after adding random noise
    so that the generated spectrograms are slightly degenerated.

    :param ratio: How many percent of spectrograms in batch to degrade (0,1)
    :param repeat: How many times to degrade
    :return:
    """
    if not ratio: return spectrograms
    if not repeat: return spectrograms

    idx = sample(range(len(spectrograms)), int(ratio * len(spectrograms)))

    with torch.no_grad():
        s = spectrograms
        for i in range(repeat):
            s, _ = model(phonemes, s, plen)

        spectrograms[idx] = s[idx]

    return spectrograms


def replace_frames_with_random(spectrograms, ratio, distrib=torch.rand):
    """

    Each spectrogram gets different frames degraded.
    To use normal noise, set distrib=lambda shape: mean + std_dev * torch.randn(x)

    :param spectrograms:
    :param ratio: between 0,1 - how many percent of frames to degrade
    :param distrib: default torch.rand -> [0, 1 uniform]
    :return:
    """
    if not ratio: return spectrograms

    t = spectrograms.shape[1]
    num_frames = int(t * ratio)
    idx = [sample(range(t), num_frames) for i in range(len(spectrograms))]  # different for each spec.

    for s, _ in enumerate(spectrograms):
        rnd_frames = distrib((num_frames, spectrograms.shape[-1])).to(spectrograms.device)
        spectrograms[s, idx[s]] = rnd_frames

    return spectrograms


def frame_dropout(spectrograms, ratio):
    """Replace random frames with zeros

    :param spectrograms:
    :param ratio:
    :return:
    """
    return replace_frames_with_random(spectrograms, ratio, distrib=lambda shape: torch.zeros(shape))


def random_patches(spectrograms1, spectrograms2, width, slen):
    """Create random patches from spectrograms

    :param spectrograms: (batch, time, channels)
    :param width: int
    :param slen: list of int
    :return: patches (batch, width, channels)
    """

    idx = [randrange(l - width) for l in slen]
    patches1, patches2 = [s[i:i+width] for s, i in zip(spectrograms1, idx)], [s[i:i+width] for s, i in zip(spectrograms2, idx)]
    return torch.stack(patches1), torch.stack(patches2)
