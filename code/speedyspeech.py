"""Train the SpeedySpeech spectrogram synthesis student model

usage: speedyspeech.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                       [--grad_clip GRAD_CLIP] [--adam_lr ADAM_LR]
                       [--standardize STANDARDIZE] [--name NAME]
                       [--durations_filename DURATIONS_FILENAME]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Training epochs
  --grad_clip GRAD_CLIP
                        Gradient clipping value
  --adam_lr ADAM_LR     Initial learning rate for adam
  --standardize STANDARDIZE
                        Standardize spectrograms
  --name NAME           Append to logdir name
  --durations_filename DURATIONS_FILENAME
                        Name for extracted dutations file
"""

import itertools

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader

import git
from barbar import Bar  # progress bar

from layers import Conv1d, ResidualBlock, FreqNorm
from losses import l1_masked, masked_huber, masked_ssim, l1_dtw
from functional import mask, positional_encoding, display_spectr_alignment
from datasets.AudioDataset import AudioDataset
from extract_durations import fert2align
from stft import MySTFT, pad_batch

from hparam import HPFertility as hp
from hparam import HPStft, HPText

from utils.transform import map_to_tensors, Pad, StandardNorm
from utils.text import TextProcessor


def expand_encodings(encodings, durations):
    """Expand phoneme encodings according to corresponding estimated durations

    Durations should be 0-masked, to prevent expanding of padded characters
    :param encodings:
    :param durations: (batch, time)
    :return:
    """
    encodings = [torch.repeat_interleave(e, d, dim=0)
                 for e, d in zip(encodings, durations.long())]

    return encodings


def expand_positional_encodings(durations, channels, repeat=False):
    """Expand positional encoding to align with phoneme durations

    Example:
        If repeat:
        phonemes a, b, c have durations 3,5,4
        The expanded encoding is
          a   a   a   b   b   b   b   b   c   c   c   c
        [e1, e2, e3, e1, e2, e3, e4, e5, e1, e2, e3, e4]

    Use Pad from transforms to get batched tensor.

    :param durations: (batch, time), 0-masked tensor
    :return: positional_encodings as list of tensors, (batch, time)
    """

    durations = durations.long()
    def rng(l): return list(range(l))

    if repeat:
        max_len = torch.max(durations)
        pe = positional_encoding(channels, max_len)
        idx = []
        for d in durations:
            idx.append(list(itertools.chain.from_iterable([rng(dd) for dd in d])))
        return [pe[i] for i in idx]
    else:
        max_len = torch.max(durations.sum(dim=-1))
        pe = positional_encoding(channels, max_len)
        return [pe[:s] for s in durations.sum(dim=-1)]


def round_and_mask(pred_durations, plen):
    pred_durations[pred_durations < 1] = 1  # we do not care about gradient outside training
    pred_durations = mask_durations(pred_durations, plen)  # the durations now expand only phonemes and not padded values
    pred_durations = torch.round(pred_durations)
    return pred_durations


def mask_durations(durations, plen):
    m = mask(durations.shape, plen, dim=-1).to(durations.device).float()
    return durations * m


class Encoder(nn.Module):
    """Encodes input phonemes for the duration predictor and the decoder"""
    def __init__(self):
        super(Encoder, self).__init__()

        self.prenet = nn.Sequential(
            nn.Embedding(hp.alphabet_size, hp.channels, padding_idx=0),
            Conv1d(hp.channels, hp.channels),
            hp.activation(),
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hp.channels, hp.enc_kernel_size, d, n=2, norm=hp.normalize, activation=hp.activation)
            for d in hp.enc_dilations
        ])

        self.post_net1 = nn.Sequential(
            Conv1d(hp.channels, hp.channels),
        )

        self.post_net2 = nn.Sequential(
            hp.activation(),
            hp.normalize(hp.channels),
            Conv1d(hp.channels, hp.channels)
        )

    def forward(self, x):

        embedding = self.prenet(x)
        x = self.res_blocks(embedding)
        x = self.post_net1(x) + embedding
        return self.post_net2(x)


class Decoder(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""
    def __init__(self):
        super(Decoder, self).__init__()

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hp.channels, hp.dec_kernel_size, d, n=2, norm=hp.normalize, activation=hp.activation)
            for d in hp.dec_dilations],
        )

        self.post_net1 = nn.Sequential(
            Conv1d(hp.channels, hp.channels),
        )

        self.post_net2 = nn.Sequential(
            ResidualBlock(hp.channels, hp.dec_kernel_size, 1, n=2),
            Conv1d(hp.channels, hp.out_channels),
            hp.final_activation()
        )

    def forward(self, x):
        xx = self.res_blocks(x)
        x = self.post_net1(xx) + x
        return self.post_net2(x)


class DurationPredictor(nn.Module):
    """Predicts phoneme log durations based on the encoder outputs"""
    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.layers = nn.Sequential(
            ResidualBlock(hp.channels, 4, 1, n=1, norm=hp.normalize, activation=nn.ReLU),
            ResidualBlock(hp.channels, 3, 1, n=1, norm=hp.normalize, activation=nn.ReLU),
            ResidualBlock(hp.channels, 1, 1, n=1, norm=hp.normalize, activation=nn.ReLU),
            Conv1d(hp.channels, 1))

    def forward(self, x):
        """Outputs interpreted as log(durations)
        To get actual durations, do exp transformation
        :param x:
        :return:
        """
        return self.layers(x)


class Interpolate(nn.Module):
    """Use multihead attention to increase variability in expanded phoneme encodings
    
    Not used in the final model, but used in reported experiments.
    """
    def __init__(self):
        super(Interpolate, self).__init__()

        ch = hp.channels
        self.att = nn.MultiheadAttention(ch, num_heads=4)
        self.norm = FreqNorm(ch)
        self.conv = Conv1d(ch, ch, kernel_size=1)

    def forward(self, x):
        xx = x.permute(1, 0, 2)  # (batch, time, channels) -> (time, batch, channels)
        xx = self.att(xx, xx, xx)[0].permute(1, 0, 2)  # (batch, time, channels)
        xx = self.conv(xx)
        return self.norm(xx) + x


class SpeedySpeech(nn.Module):
    """The SpeedySpeech student model"""
    def __init__(
            self,
            adam_lr=0.02,
            standardize=True,
            device='cuda',
            durations_file='durations.txt'
    ):
        super(SpeedySpeech, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.duration_predictor = DurationPredictor()
        self.pad = Pad(0)
        if hp.interpolate:
            self.interpolate = Interpolate()

        # collate function
        self.collate = Collate(device=device, standardize=standardize)

        # optim
        self.optimizer = torch.optim.Adam(self.parameters(), lr=adam_lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3)

        # losses
        self.loss_l1 = l1_masked
        self.loss_ssim = masked_ssim
        self.loss_huber = masked_huber

        # device
        self.device=device
        self.to(self.device)
        print(f'Model sent to {self.device}')

        # helper vars
        self.checkpoint = None
        self.epoch = 0
        self.step = 0
        self.durations_file = durations_file

        repo = git.Repo(search_parent_directories=True)
        self.git_commit = repo.head.object.hexsha

    def forward(self, x):
        """

        To get estimated length for each spectrogram, run durations.sum(axis=-1)
        On training, log(pred_durations) is returned, on inference, pred_durations is returned

        :param x: (phonemes, plen, durations) or (phonemes, plen)
            phonemes: 2D tensor of padded index phonemes
            plen: list of phoneme lengths before padding
            durations: 2D zero-padded tensor of phoneme durations, one row per sentence
        :returns: (spectrograms, log(pred_durations)) or (spectrograms, pred_durations)
        """

        if self.training:
            phonemes, plen, durations = x
        else:
            phonemes, plen = x

        encodings = self.encoder(phonemes)  # batch, time, channels
        pred_durations = self.duration_predictor(encodings.detach() if hp.separate_duration_grad
                                                 else encodings)[..., 0]  # batch, time

        # use exp(log(durations)) = durations
        if not self.training:
            pred_durations = round_and_mask(torch.exp(pred_durations), plen)
            encodings = self.expand_enc(encodings, pred_durations)
        else:
            encodings = self.expand_enc(encodings, durations)

        if hp.interpolate:
            encodings = self.interpolate(encodings)
        decoded = self.decoder(encodings)
        return decoded, pred_durations

    # todo: make functional?
    def expand_enc(self, encodings, durations):
        """Copy each phoneme encoding as many times as the duration predictor predicts"""
        encodings = self.pad(expand_encodings(encodings, durations))
        if hp.pos_enc:
            if hp.pos_enc == 'ours':
                encodings += self.pad(expand_positional_encodings(durations, encodings.shape[-1])).to(encodings.device)
            elif hp.pos_enc == 'standard':
                encodings += positional_encoding(encodings.shape[-1], encodings.shape[1]).to(encodings.device)
        return encodings

    def save(self):

        if self.checkpoint is not None:
            os.remove(self.checkpoint)
        self.checkpoint = os.path.join(self.logger.log_dir, f'{time.strftime("%Y-%m-%d")}_checkpoint_step{self.step}.pth')
        torch.save(
            {
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'git_commit': self.git_commit
            },
            self.checkpoint)

    def load(self, checkpoint, map_location=False):
        if map_location:
            checkpoint = torch.load(checkpoint, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint)
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        commit = checkpoint['git_commit']
        if commit != self.git_commit:
            print(f'Warning: the loaded checkpoint was trained on commit {commit}, but you are on {self.git_commit}')
        self.checkpoint = None  # prevent overriding old checkpoint
        return self

    def fit(self, batch_size, logdir, epochs=1, grad_clip=1, checkpoint_every=10):
        self.grad_clip = grad_clip
        self.logger = SummaryWriter(logdir)

        train_loader = self.train_dataloader(batch_size)
        valid_loader = self.val_dataloader(batch_size)

        # continue training from self.epoch if checkpoint loaded
        for e in range(self.epoch + 1, self.epoch + 1 + epochs):
            self.epoch = e
            train_losses = self._train_epoch(train_loader)
            valid_losses = self._validate(valid_loader)

            self.scheduler.step(sum(valid_losses))
            self.logger.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)

            if not e % checkpoint_every:
                self.save()

            print(f'Epoch {e} | Train - l1: {train_losses[0]}, ssim: {train_losses[1]}, log_huber: {train_losses[2]}| '
                  f'Valid - l1: {valid_losses[0]}, log_huber: {valid_losses[1]}|')

    def _train_epoch(self, dataloader):
        self.train()

        t_l1, t_ssim, t_huber = 0,0,0
        for i, batch in enumerate(Bar(dataloader)):
            self.optimizer.zero_grad()
            spectrs, slen, phonemes, plen, text, durations = batch
            out, pred_durations = self.forward((phonemes, plen, durations))

            l1 = self.loss_l1(out, spectrs, slen)
            ssim = self.loss_ssim(out, spectrs, slen)
            durations[durations < 1] = 1  # needed to prevent log(0)
            huber = self.loss_huber(pred_durations, torch.log(durations.float()), plen)

            loss = l1 + ssim + huber
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()
            self.step += 1

            t_l1 += l1.item()
            t_ssim += ssim.item()
            t_huber += huber.item()

            self.logger.add_scalar(
                'batch/total', loss.item(), self.epoch * len(dataloader) + i
            )

        # report average cost per batch
        self.logger.add_scalar('train/l1', t_l1 / i, self.epoch)
        self.logger.add_scalar('train/ssim', t_ssim / i, self.epoch)
        self.logger.add_scalar('train/log(durations)_huber', t_huber / i, self.epoch)
        return t_l1, t_ssim, t_huber

    def _validate(self, dataloader):
        self.eval()

        t_l1, t_huber = 0,0
        for i, batch in enumerate(dataloader):
            spectrs, slen, phonemes, plen, text, durations = batch
            with torch.no_grad():
                out, pred_durations = self.forward((phonemes, plen))

            estimated_slen = pred_durations.sum(axis=-1).long()
            l1 = l1_dtw(spectrs, slen, out, estimated_slen)
            huber = self.loss_huber(pred_durations, durations.float(), plen)  # durations on linear scale

            t_l1 += l1
            t_huber += huber.item()

            durations = fert2align(durations.cpu().numpy())
            durations = [torch.as_tensor(d) for d in durations]
            pred_durations = fert2align(pred_durations.cpu().numpy())
            pred_durations = [torch.as_tensor(d) for d in pred_durations]

            fig = display_spectr_alignment(out[-1, :estimated_slen[-1]],
                                           pred_durations[-1][:estimated_slen[-1], :plen[-1]],
                                           spectrs[-1, :slen[-1]], durations[-1][:slen[-1], :plen[-1]],
                                           text[-1])
            self.logger.add_figure(text[-1], fig, self.epoch)

            # log audio every 10 epochs
            if not self.epoch % 10:
                spec = self.collate.norm.inverse(out[-1:]) # TODO: this fails if we do not standardize!
                sound, length = self.collate.stft.spec2wav(spec.transpose(1, 2), estimated_slen[-1:])
                sound = sound[0, :length[0]]
                self.logger.add_audio(text[-1], sound.detach().cpu().numpy(), self.epoch, sample_rate=22050) # TODO: parameterize

        # report average cost per batch
        self.logger.add_scalar('valid/l1', t_l1 / i, self.epoch)
        self.logger.add_scalar('valid/durations_huber', t_huber / i, self.epoch)
        return t_l1, t_huber

    def train_dataloader(self, batch_size):
        return DataLoader(AudioDataset(HPText.dataset, start_idx=0, end_idx=HPText.num_train, durations=self.durations_file), batch_size=batch_size,
                          collate_fn=self.collate,
                          shuffle=True)

    def val_dataloader(self, batch_size):
        dataset = AudioDataset(HPText.dataset, start_idx=HPText.num_train, end_idx=HPText.num_valid, durations=self.durations_file)
        return DataLoader(dataset, batch_size=batch_size,
                          collate_fn=self.collate,
                          shuffle=False, sampler=SequentialSampler(dataset))


class Collate(nn.Module):
    def __init__(self, device, standardize=False):
        super(Collate, self).__init__()
        self.device = device
        self.text_proc = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
        self.stft = MySTFT().to(device)
        self.norm = StandardNorm(mean=HPStft.spec_mean, std=HPStft.spec_std).to(device) if standardize else nn.Identity()

    def __call__(self, list_of_tuples):
        text, wav, alignments = list(zip(*list_of_tuples))

        align = [torch.as_tensor(x) for x in alignments]
        align = pad_batch(align)[0].to(self.device)

        phonemes, plen = self.text_proc(text)
        phonemes = phonemes.to(self.device)

        spectrs, slen = self.stft.wav2spec(wav)
        spectrs = self.norm(spectrs)
        spectrs = spectrs.transpose(2,1).to(self.device)
        return spectrs, slen, phonemes, plen, text, align



if __name__ == '__main__':
    import argparse, os, time
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--epochs", default=100, type=int, help="Training epochs")
    parser.add_argument("--grad_clip", default=1, type=int, help="Gradient clipping value")
    parser.add_argument("--adam_lr", default=0.002, type=int, help="Initial learning rate for adam")
    parser.add_argument("--standardize", default=True, type=bool, help="Standardize spectrograms")
    parser.add_argument("--name", default="", type=str, help="Append to logdir name")
    parser.add_argument("--durations_filename", default="durations.txt", type=str, help="Name for extracted dutations file")
    args = parser.parse_args()

    m = SpeedySpeech(
        adam_lr=args.adam_lr,
        standardize=args.standardize,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        durations_file=args.durations_filename
    )

    m.fit(
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        checkpoint_every=10,
        logdir=os.path.join('logs2', time.strftime("%Y-%m-%dT%H-%M-%S") + '-' + args.name)
    )
