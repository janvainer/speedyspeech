import torch
import torch.nn.functional as F
from librosa.feature.inverse import mel_to_stft
from librosa.filters import mel as librosa_mel_fn
import numpy as np
from utils.torch_stft import STFT
from functional import pad_batch
from hparam import HPStft as hp


def mel_to_stft_torch(mel_spectrs, n_fft):
    return torch.stack([torch.as_tensor(mel_to_stft(s.cpu().numpy(), n_fft=n_fft, power=1.0))
                        for s in mel_spectrs]).to(mel_spectrs.device)


def dynamic_range_compression(x, C=1, clip_val=hp.clip_val):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    return torch.exp(x) / C


def nnls(mel_basis, mel_spec, n_iter=5, max_iter_per_step=20, lr=2):
    """Minimize sum_i, j((AX)[i, j] - B[i, j]) ^ 2

    :param mel_basis: mel-basis (n_mels, n_fft // 2 +1)
    :param mel_spec: linear-mel-spectrogram (batch, n_mels, time)
    :return: X: linear spectrogram (batch, n_fft // 2 +1, time), non-negative
    """

    inv = torch.pinverse(mel_basis)
    X = torch.matmul(inv, mel_spec)
    X.requires_grad = True
    optim = torch.optim.LBFGS([X], lr, max_iter=max_iter_per_step, line_search_fn='strong_wolfe')

    for i in range(n_iter):
        def closure():
            optim.zero_grad()
            l = torch.matmul(mel_basis, X) - mel_spec
            # regularize to get positive values
            loss = torch.sum(torch.pow(l, 2.0)) + 0.1*X[X<0].abs().sum()
            loss.backward()
            return loss

        torch.nn.utils.clip_grad_norm_(X, 1)
        optim.step(closure)
    return X.abs()


class MySTFT(torch.nn.Module):
    def __init__(self,
            n_fft=hp.n_fft,  # filter length
            hop_length=hp.hop_size,
            win_length=hp.win_size,
            sampling_rate=hp.sample_rate,
            n_mel=hp.n_mel,
            mel_fmin=hp.mel_fmin,
            mel_fmax=hp.mel_fmax):
        super(MySTFT, self).__init__()

        self.n_mel_channels = n_mel
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        # does reflection padding with n_fft // 2 on both sides of signal
        self.stft = STFT(n_fft, hop_length, win_length, window='hann')

        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        mel_inverse = torch.pinverse(mel_basis)
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('mel_inverse', mel_inverse)


    def wav2spec(self, y):
        """log-mel-spectrograms

        :param y: list of unpadded wavs
        :param lens:
        :return: (batch, freq_bins, time), list_of_lens
        """

        # + added by stft, * actual sound, - added by us, 0 zero pad
        # -> eliminate the + at the end = n_fft // (2 * hop_length) frames
        #[++++][**********][----][++++]
        #[++++][*****][----]00000[++++]
        lens = [len(yy) for yy in y]
        y = [F.pad(torch.as_tensor(yy[None, None, :]), pad=(0, self.n_fft // 2), mode='reflect').squeeze()
             for yy in y]
        y, _ = pad_batch(y, 0)
        y = y.to(self.mel_basis.device)

        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)

        slen = [l // self.hop_length + 1 for l in lens]
        magnitudes, phases = self.stft.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = dynamic_range_compression(mel_output)

        to_drop = self.n_fft // (2 * self.hop_length)
        return mel_output[..., :-to_drop], slen

    def spec2wav(self, log_mel, lens, n_iters=30):
        """Invert log-mel spectrograms to sound.

        Use mel filterbank inversion from librosa and griffin-lim for phase estimation.
        Apply iSTFT to the result.

        :param log_mel: (batch, freq_bins, time)
        :param n_iters:
        :return:
        """
        magnitudes = dynamic_range_decompression(log_mel)
        magnitudes = self.mel2linear(magnitudes)
        signal = self.griffin_lim(magnitudes, n_iters=n_iters)

        lens = [self.hop_length * l for l in lens]
        return signal, lens

    def griffin_lim(self, magnitudes, n_iters=30):
        angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
        angles = angles.astype(np.float32)
        angles = torch.autograd.Variable(torch.from_numpy(angles)).to(magnitudes.device)
        signal = self.stft.inverse(magnitudes, angles).squeeze(1)

        for i in range(n_iters):
            _, angles = self.stft.transform(signal)
            signal = self.stft.inverse(magnitudes, angles).squeeze(1)
        return signal

    def mel2linear(self, mel):
        return nnls(self.mel_basis, mel)
