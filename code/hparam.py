import os

import torch
from layers import FreqNorm
from utils.text import TextProcessor

class HPStft:
    sample_rate = 22050
    n_mel = 80
    n_fft = 1024
    hop_size = 256
    win_size = 1024
    mel_fmin = 0.0
    mel_fmax = 8000.0
    clip_val = 1e-5  # every magnitude under this value is clipped - used in dynamic range compression

    # precomputed statistics for log-mel-spectrs for LJSpeech
    spec_mean = -5.522
    spec_std = 2.063
    spec_min = -11.5129
    spec_max = 2.0584


class HPText:
    # needed to make independent on the directory from which python is invoked
    dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/data/LJSpeech-1.1')
    num_train, num_valid = 13000, 13099  # train will use (0, 13000), valid wil use (13000, 13099)

    punctuation = list("'\",.:?!")
    graphemes = ["<pad>", "<unk>"] + list('abcdefghijklmnopqrstuvwxyz ') + punctuation
    # graphemes_czech = ["<pad>", "<unk>"] + list('aábcčdďeěfghiíjklmnňoópqrřsštťuůúvwxyýzž ') + punctuation
    use_phonemes = True

class HPFertility:

    separate_duration_grad = True
    out_channels = HPStft.n_mel
    alphabet_size = len(TextProcessor.phonemes) if HPText.use_phonemes else len(HPText.graphemes)
    channels = 128
    enc_kernel_size = 4
    dec_kernel_size = 4
    enc_dilations = 4 * [1,2,4] + [1]  # receptive field is max 15
    dec_dilations = 4 * [1,2,4,8] + [1]  # receptive field is max 32
    normalize = FreqNorm
    activation = torch.nn.ReLU
    final_activation = torch.nn.Identity
    pos_enc = 'ours'  # False, 'standard', 'ours'
    interpolate = False # True


class HPDurationExtractor:
    positional_encoding = True
    w = 6.42
    sigma = 0.3  # weight for guided attention
    scale_min, scale_max = 0, 1  # mel_spectr values will be scaled to this interval

    alphabet_size = len(TextProcessor.phonemes) if HPText.use_phonemes else len(HPText.graphemes)
    channels = 40
    hidden_channels = 80
    kernel_size = 3
    out_channels = HPStft.n_mel
    out_activation = torch.nn.Sigmoid
    nonlinearity = torch.nn.ReLU

    dilations_txt_enc = 2 * [3 ** i for i in range(4)] + [1, 1]
    dilations_audio_enc = 2 * [3 ** i for i in range(4)] + [1, 1]
    dilations_dec = 2*[3 ** i for i in range(4)] + [1, 1]

    att_noise = 0.1
    att_hidden_channels = 80

    # Spectrogram augmentation
    # 1. add normal noise to input spectrs
    noise = 0.01
    # 2. Feed spectrograms through the model `feed_repeat` times
    # use degraded output on input for training
    feed_repeat = 2
    feed_ratio = 0.5  # how many items in batch are degraded
    # 3. Replace random spectrogram frames with random noise
    replace_ratio = 0.1
