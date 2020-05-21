import torch
import numpy as np
import matplotlib.pyplot as plt


def scaled_dot_attention(q, k, v, mask=None, noise=0, dropout=lambda x: x):
    """
    :param q: queries, (batch, time1, channels1)
    :param k: keys, (batch, time2, channels1)
    :param v: values, (batch, time2, channels2)
    :param mask: boolean mask, (batch, time1, time2)
    :param dropout: a dropout function - this allows keeping dropout as a module -> better control when training/eval
    :return: (batch, time1, channels2), (batch, time1, time2)
    """

    # (batch, time1, time2)
    weights = torch.matmul(q, k.transpose(2, 1))
    if mask is not None:
        weights = weights.masked_fill(~mask, float('-inf'))

    if noise:
        weights += noise * torch.randn(weights.shape).to(weights.device)

    weights = torch.softmax(weights, dim=-1)
    weights = dropout(weights)

    result = torch.matmul(weights, v)  # (batch, time1, channels2)
    return result, weights


def positional_encoding(channels, length, w=1):
    """The positional encoding from `Attention is all you need` paper

    :param channels: How many channels to use
    :param length: 
    :param w: Scaling factor
    :return:
    """
    enc = torch.FloatTensor(length, channels)
    rows = torch.arange(length, out=torch.FloatTensor())[:, None]
    cols = 2 * torch.arange(channels//2, out=torch.FloatTensor())

    enc[:, 0::2] = torch.sin(w * rows / (10.0**4 ** (cols / channels)))
    enc[:, 1::2] = torch.cos(w * rows / (10.0**4 ** (cols / channels)))
    return enc


def median(distribution, keepdim=True):
    cum_dist = torch.cumsum(distribution, dim=-1)
    return torch.sum(cum_dist < 0.5, -1, keepdim=keepdim).float()


def median_mask(distribution, window=(5,5)):
    """Expects distribution of shape (batch, time, channels)"""
    med = median(distribution)
    m = torch.ones(distribution.shape).float() * torch.arange(distribution.shape[-1]).float()
    m = m.to(distribution.device)
    mask = (med - window[0] <= m) * (m <= med + window[1])
    return mask


def idx_mask(shape, idx, size):
    """

    :param distribution: (batch, time, channels)
    :param idx: (batch, len_spectrograms, 1) eg torch.argmax(weights).unsqueeze(-1)
    :param size:
    :return:
    """
    m = torch.ones(shape).float() * torch.arange(shape[-1]).float()
    m = m.to(idx.device)
    mask = (idx <= m) * (m <= idx + size)
    return mask


def mask(shape, lengths, dim=-1):

    assert dim != 0, 'Masking not available for batch dimension'
    assert len(lengths) == shape[0], 'Lengths must contain as many elements as there are items in the batch'

    lengths = torch.as_tensor(lengths)

    to_expand = [1] * (len(shape)-1)+[-1]
    mask = torch.arange(shape[dim]).expand(to_expand).transpose(dim, -1).expand(shape).to(lengths.device)
    mask = mask < lengths.expand(to_expand).transpose(0, -1)
    return mask


def get_phoneme_durations(alignment):
    """Alignment must be a batch

    :return counts: [(idx, count), ...]
    """

    alignment = torch.as_tensor(alignment)
    maxx = torch.max(alignment, dim=-1)[1]
    counts = [torch.unique(m, return_counts=True) for m in maxx]

    return counts


def display_spectr_alignment(spec_supervised, align_supervised, spec_generated, align_generated, text):
    """

    :param spectrogram_supervised: (time, channels), expected on gpu, no grad attached
    :param alignment: (time, phonemes)
    :return: figure
    """
    s1 = spec_generated.detach().cpu().numpy().T
    a1 = align_generated.detach().cpu().numpy().T
    m1 = np.argmax(a1, axis=0) + s1.shape[0]
    sa1 = np.concatenate((s1, a1), axis=0)

    s2 = spec_supervised.detach().cpu().numpy().T
    a2 = align_supervised.detach().cpu().numpy().T
    m2 = np.argmax(a2, axis=0) + s2.shape[0]
    sa2 = np.concatenate((s2, a2), axis=0)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(text)
    ax[0].imshow(sa1, interpolation='none')
    ax[0].xaxis.set_ticks_position('bottom')
    ax[0].plot(m1, color='red', linestyle='--', alpha=0.7)

    ax[1].imshow(sa2, interpolation='none')
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].plot(m2, color='red', linestyle='--', alpha=0.7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def pad_batch(items, pad_value=0):
    """Pad tensors in list to equal length (in the first dim)

    :param items:
    :param pad_value:
    :return: padded_items, orig_lens
    """
    max_len = len(max(items, key=lambda x: len(x)))
    zeros = (2*torch.as_tensor(items[0]).ndim -1) * [pad_value]
    padded_items = torch.stack([torch.nn.functional.pad(torch.as_tensor(x), pad= zeros + [max_len - len(x)], value=pad_value)
                          for x in items])
    orig_lens = [len(xx) for xx in items]
    return padded_items, orig_lens
