"""The duration extraction teacher model

Run this script to train the duration extraction model.

usage: duration_extractor.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                             [--grad_clip GRAD_CLIP] [--adam_lr ADAM_LR]
                             [--warmup_epochs WARMUP_EPOCHS]
                             [--from_checkpoint FROM_CHECKPOINT] [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Training epochs
  --grad_clip GRAD_CLIP
                        Gradient clipping value
  --adam_lr ADAM_LR     Initial learning rate for adam
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs for NoamScheduler
  --from_checkpoint FROM_CHECKPOINT
                        Checkpoint file path
  --name NAME           Append to logdir name
"""

import torch.nn as nn                     # neural networks
from torch.nn import L1Loss, ZeroPad2d
from torch.utils.tensorboard import SummaryWriter
import torch

import git
from barbar import Bar  # progress bar

from layers import WaveResidualBlock, Conv1d
from functional import positional_encoding, median_mask, mask, idx_mask, scaled_dot_attention, display_spectr_alignment
from losses import l1_masked, GuidedAttentionLoss

from hparam import HPDurationExtractor as hp
from hparam import HPStft, HPText

from utils.augment import add_random_noise, degrade_some, frame_dropout
from utils.optim import NoamScheduler
from utils.transform import MinMaxNorm
from utils.text import TextProcessor

from datasets.AudioDataset import AudioDataset
from torch.utils.data import DataLoader

from stft import MySTFT, pad_batch
from torch.utils.data.sampler import SequentialSampler

class ConvTextEncoder(nn.Module):
    """Encodes input phonemes into keys and values"""
    
    def __init__(self):
        super(ConvTextEncoder, self).__init__()

        self.embedding = nn.Embedding(hp.alphabet_size, hp.channels, padding_idx=0)  # padding idx mapped to zero vector
        layers = [Conv1d(hp.channels, hp.channels),
                  hp.nonlinearity()]

        layers.extend([
            WaveResidualBlock(hp.channels, hp.hidden_channels, hp.kernel_size, d, causal=False)
            for d in hp.dilations_txt_enc
        ])

        layers.append(Conv1d(hp.channels, hp.channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        emb = self.embedding(x)
        keys = self.layers(emb)
        values = (keys + emb) * torch.sqrt(torch.as_tensor(0.5))  # TODO: try without this
        return keys, values


class ConvAudioEncoder(nn.Module):
    """Encodes input spectrograms into queries"""
    def __init__(self):
        super(ConvAudioEncoder, self).__init__()

        layers = [Conv1d(hp.out_channels, hp.channels),
                  hp.nonlinearity()]

        layers.extend([
            WaveResidualBlock(hp.channels, hp.hidden_channels, hp.kernel_size, d, causal=True)
            for d in hp.dilations_audio_enc
        ])

        self.layers = nn.Sequential(*layers)

    def generating(self, mode):
        """Put the module into mode for sequential generation"""
        # reset queues
        for module in self.layers.children():
            if hasattr(module, 'generating'):
                module.generating(mode)

    def forward(self, x):
        return self.layers(x)


class ConvAudioDecoder(nn.Module):
    """Decodes result of attention layer into spectrogram"""
    def __init__(self):
        super(ConvAudioDecoder, self).__init__()

        layers =[
            WaveResidualBlock(hp.channels, hp.hidden_channels, hp.kernel_size, d, causal=True)
            for d in hp.dilations_dec
        ]

        layers.extend([
            Conv1d(hp.channels, hp.channels),
            hp.nonlinearity(),
            Conv1d(hp.channels, hp.channels),
            hp.nonlinearity(),
            Conv1d(hp.channels, hp.channels),
            hp.nonlinearity(),
            Conv1d(hp.channels, hp.out_channels),
            hp.out_activation()
        ])
        self.layers = nn.Sequential(*layers)

    def generating(self, mode):
        """Put the module into mode for sequential generation"""
        for module in self.layers.children():
            if hasattr(module, 'generating'):
                module.generating(mode)

    def forward(self, x):
        return self.layers(x)


class ScaledDotAttention(nn.Module):
    """Scaled dot attention with positional encoding preconditioning"""

    def __init__(self):
        super(ScaledDotAttention, self).__init__()

        self.noise = hp.att_noise
        self.fc_query = Conv1d(hp.channels, hp.att_hidden_channels)
        self.fc_keys = Conv1d(hp.channels, hp.att_hidden_channels)

        # share parameters
        self.fc_keys.weight = torch.nn.Parameter(self.fc_query.weight.clone())
        self.fc_keys.bias = torch.nn.Parameter(self.fc_query.bias.clone())

        self.fc_values = Conv1d(hp.channels, hp.att_hidden_channels)
        self.fc_out = Conv1d(hp.att_hidden_channels, hp.channels)

    def forward(self, q, k, v, mask=None):
        """
        :param q: queries, (batch, time1, channels1)
        :param k: keys, (batch, time2, channels1)
        :param v: values, (batch, time2, channels2)
        :param mask: boolean mask, (batch, time1, time2)
        :return: (batch, time1, channels2), (batch, time1, time2)
        """

        noise = self.noise if self.training else 0

        alignment, weights = scaled_dot_attention(self.fc_query(q),
                                                  self.fc_keys(k),
                                                  self.fc_values(v),
                                                  mask, noise=noise)
        alignment = self.fc_out(alignment)
        return alignment, weights


class DurationExtractor(nn.Module):
    """The teacher model for duration extraction"""
    def __init__(
            self,
            adam_lr=0.002,
            warmup_epochs=30,
            init_scale=0.25,
            guided_att_sigma=0.3,
            device='cuda'
    ):
        super(DurationExtractor, self).__init__()

        self.txt_encoder = ConvTextEncoder()
        self.audio_encoder = ConvAudioEncoder()
        self.audio_decoder = ConvAudioDecoder()
        self.attention = ScaledDotAttention()
        self.collate = Collate(device=device)

        # optim
        self.optimizer = torch.optim.Adam(self.parameters(), lr=adam_lr)
        self.scheduler = NoamScheduler(self.optimizer, warmup_epochs, init_scale)

        # losses
        self.loss_l1 = l1_masked
        self.loss_att = GuidedAttentionLoss(guided_att_sigma)

        # device
        self.device=device
        self.to(self.device)
        print(f'Model sent to {self.device}')

        # helper vars
        self.checkpoint = None
        self.epoch = 0
        self.step = 0

        repo = git.Repo(search_parent_directories=True)
        self.git_commit = repo.head.object.hexsha

    def to_device(self, device):
        print(f'Sending network to {device}')
        self.device = device
        self.to(device)
        return self

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
                'scheduler': self.scheduler.state_dict(),
                'git_commit': self.git_commit
            },
            self.checkpoint)

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        commit = checkpoint['git_commit']
        if commit != self.git_commit:
            print(f'Warning: the loaded checkpoint was trained on commit {commit}, but you are on {self.git_commit}')
        self.checkpoint = None  # prevent overriding old checkpoint
        return self

    def forward(self, phonemes, spectrograms, len_phonemes, training=False):
        """
        :param phonemes: (batch, alphabet, time), padded phonemes
        :param spectrograms: (batch, freq, time), padded spectrograms
        :param len_phonemes: list of phoneme lengths
        :return: decoded_spectrograms, attention_weights
        """
        spectrs = ZeroPad2d((0,0,1,0))(spectrograms)[:, :-1, :]  # move this to encoder?
        keys, values = self.txt_encoder(phonemes)
        queries = self.audio_encoder(spectrs)

        att_mask = mask(shape=(len(keys), queries.shape[1], keys.shape[1]),
                        lengths=len_phonemes,
                        dim=-1).to(self.device)

        if hp.positional_encoding:
            keys += positional_encoding(keys.shape[-1], keys.shape[1], w=hp.w).to(self.device)
            queries += positional_encoding(queries.shape[-1], queries.shape[1], w=1).to(self.device)

        attention, weights = self.attention(queries, keys, values, mask=att_mask)
        decoded = self.audio_decoder(attention + queries)
        return decoded, weights

    def generating(self, mode):
        """Put the module into mode for sequential generation"""
        for module in self.children():
            if hasattr(module, 'generating'):
                module.generating(mode)

    def generate(self, phonemes, len_phonemes, steps=False, window=3, spectrograms=None):
        """Sequentially generate spectrogram from phonemes
        
        If spectrograms are provided, they are used on input instead of self-generated frames (teacher forcing)
        If steps are provided with spectrograms, only 'steps' frames will be generated in supervised fashion
        Uses layer-level caching for faster inference.

        :param phonemes: Padded phoneme indices
        :param len_phonemes: Length of each sentence in `phonemes` (list of lengths)
        :param steps: How many steps to generate
        :param window: Window size for attention masking
        :param spectrograms: Padded spectrograms
        :return: Generated spectrograms
        """
        self.generating(True)
        self.train(False)

        assert steps or (spectrograms is not None)
        steps = steps if steps else spectrograms.shape[1]

        with torch.no_grad():
            phonemes = torch.as_tensor(phonemes)
            keys, values = self.txt_encoder(phonemes)

            if hp.positional_encoding:
                keys += positional_encoding(keys.shape[-1], keys.shape[1], w=hp.w).to(self.device)
                pe = positional_encoding(hp.channels, steps, w=1).to(self.device)

            if spectrograms is None:
                dec = torch.zeros(len(phonemes), 1, hp.out_channels, device=self.device)
            else:
                input = ZeroPad2d((0, 0, 1, 0))(spectrograms)[:, :-1, :]

            weights, decoded = None, None

            if window is not None:
                shape = (len(phonemes), 1, phonemes.shape[-1])
                idx = torch.zeros(len(phonemes), 1, phonemes.shape[-1]).to(phonemes.device)
                att_mask = idx_mask(shape, idx, window)
            else:
                att_mask = mask(shape=(len(phonemes), 1, keys.shape[1]),
                                lengths=len_phonemes,
                                dim=-1).to(self.device)

            for i in range(steps):
                if spectrograms is None:
                    queries = self.audio_encoder(dec)
                else:
                    queries = self.audio_encoder(input[:, i:i+1, :])

                if hp.positional_encoding:
                    queries += pe[i]

                att, w = self.attention(queries, keys, values, att_mask)
                dec = self.audio_decoder(att + queries)
                weights = w if weights is None else torch.cat((weights, w), dim=1)
                decoded = dec if decoded is None else torch.cat((decoded, dec), dim=1)
                if window is not None:
                    idx = torch.argmax(w, dim=-1).unsqueeze(2).float()
                    att_mask = idx_mask(shape, idx, window)

        self.generating(False)
        return decoded, weights

    def generate_naive(self, phonemes, len_phonemes, steps=1, window=(0,1)):
        """Naive generation without layer-level caching for testing purposes"""
                                       
        self.train(False)

        with torch.no_grad():
            phonemes = torch.as_tensor(phonemes)

            keys, values = self.txt_encoder(phonemes)

            if hp.positional_encoding:
                keys += positional_encoding(keys.shape[-1], keys.shape[1], w=hp.w).to(self.device)
                pe = positional_encoding(hp.channels, steps, w=1).to(self.device)

            dec = torch.zeros(len(phonemes), 1, hp.out_channels, device=self.device)

            weights = None

            att_mask = mask(shape=(len(phonemes), 1, keys.shape[1]),
                            lengths=len_phonemes,
                            dim=-1).to(self.device)

            for i in range(steps):
                print(i)
                queries = self.audio_encoder(dec)
                if hp.positional_encoding:
                    queries += pe[i]

                att, w = self.attention(queries, keys, values, att_mask)
                d = self.audio_decoder(att + queries)
                d = d[:, -1:]
                w = w[:, -1:]
                weights = w if weights is None else torch.cat((weights, w), dim=1)
                dec = torch.cat((dec, d), dim=1)

                if window is not None:
                    att_mask = median_mask(weights, window=window)

        return dec[:, 1:, :], weights

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

            self.scheduler.step()
            self.logger.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)
            if not e % checkpoint_every:
                self.save()

            print(f'Epoch {e} | Train - l1: {train_losses[0]}, guided_att: {train_losses[1]}| '
                  f'Valid - l1: {valid_losses[0]}, guided_att: {valid_losses[1]}|')

    def _train_epoch(self, dataloader):
        self.train()

        t_l1, t_att = 0, 0
        for i, batch in enumerate(Bar(dataloader)):
            self.optimizer.zero_grad()
            spectrs, slen, phonemes, plen, text = batch

            s = add_random_noise(spectrs, hp.noise)
            s = degrade_some(self, s, phonemes, plen, hp.feed_ratio, repeat=hp.feed_repeat)
            s = frame_dropout(s, hp.replace_ratio)

            out, att_weights = self.forward(phonemes, s, plen)

            l1 = self.loss_l1(out, spectrs, slen)
            l_att = self.loss_att(att_weights, slen, plen)

            loss = l1 + l_att
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()
            self.step += 1

            t_l1 += l1.item()
            t_att += l_att.item()

            self.logger.add_scalar(
                'batch/total', loss.item(), self.step
            )

        # report average cost per batch
        self.logger.add_scalar('train/l1', t_l1 / i, self.epoch)
        self.logger.add_scalar('train/guided_att', t_att / i, self.epoch)
        return t_l1 / i, t_att / i

    def _validate(self, dataloader):
        self.eval()

        t_l1, t_att = 0,0
        for i, batch in enumerate(dataloader):
            spectrs, slen, phonemes, plen, text = batch
            # generate sequentially
            out, att_weights = self.generate(phonemes, plen, steps=spectrs.shape[1], window=None)

            # generate in supervised fashion - for visualisation only
            with torch.no_grad():
                out_s, att_s = self.forward(phonemes, spectrs, plen)

            l1 = self.loss_l1(out, spectrs, slen)
            l_att = self.loss_att(att_weights, slen, plen)
            t_l1 += l1.item()
            t_att += l_att.item()

            fig = display_spectr_alignment(out[-1, :slen[-1]],
                                           att_weights[-1][:slen[-1], :plen[-1]],
                                           out_s[-1, :slen[-1]], att_s[-1][:slen[-1], :plen[-1]],
                                           text[-1])
            self.logger.add_figure(text[-1], fig, self.epoch)

            if not self.epoch % 10:
                spec = self.collate.norm.inverse(out[-1:]) # TODO: this fails if we do not standardize!
                sound, length = self.collate.stft.spec2wav(spec.transpose(1, 2), slen[-1:])
                sound = sound[0, :length[0]]
                self.logger.add_audio(text[-1], sound.detach().cpu().numpy(), self.epoch, sample_rate=22050) # TODO: parameterize

        # report average cost per batch
        self.logger.add_scalar('valid/l1', t_l1 / i, self.epoch)
        self.logger.add_scalar('valid/guided_att', t_att / i, self.epoch)
        return t_l1/i, t_att/i

    def train_dataloader(self, batch_size):
        return DataLoader(AudioDataset(HPText.dataset, start_idx=0, end_idx=HPText.num_train, durations=False), batch_size=batch_size,
                          collate_fn=self.collate,
                          shuffle=True)

    def val_dataloader(self, batch_size):
        dataset = AudioDataset(HPText.dataset, start_idx=HPText.num_train, end_idx=HPText.num_valid, durations=False)
        return DataLoader(dataset, batch_size=batch_size,
                          collate_fn=self.collate,
                          shuffle=False, sampler=SequentialSampler(dataset))


class Collate:
    def __init__(self, device):
        self.device = device
        self.text_proc = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
        self.stft = MySTFT().to(device)
        self.norm = MinMaxNorm(HPStft.spec_min, HPStft.spec_max, hp.scale_min, hp.scale_max)  # scale log-mel-spectrs some interval

    def __call__(self, list_of_tuples):
        text, wav, _ = list(zip(*list_of_tuples))

        phonemes, plen = self.text_proc(text)
        phonemes = phonemes.to(self.device)

        spectrs, slen = self.stft.wav2spec(wav)
        spectrs = self.norm(spectrs)
        spectrs.clamp_(hp.scale_min,hp.scale_max)
        spectrs = spectrs.transpose(2,1).to(self.device)

        return spectrs, slen, phonemes, plen, text


if __name__ == '__main__':
    import argparse, os, time
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--epochs", default=300, type=int, help="Training epochs")
    parser.add_argument("--grad_clip", default=1, type=int, help="Gradient clipping value")
    parser.add_argument("--adam_lr", default=0.002, type=int, help="Initial learning rate for adam")
    parser.add_argument("--warmup_epochs", default=30, type=int, help="Warmup epochs for NoamScheduler")
    parser.add_argument("--from_checkpoint", default=False, type=str, help="Checkpoint file path")
    parser.add_argument("--name", default="", type=str, help="Append to logdir name")
    args = parser.parse_args()

    m = DurationExtractor(
        adam_lr=0.002,
        warmup_epochs=30,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    logdir = os.path.join('logs', time.strftime("%Y-%m-%dT%H-%M-%S") + '-' + args.name)
    if args.from_checkpoint:
        m.load(args.from_checkpoint)
        # use the folder with checkpoint as a logdir
        logdir = os.path.dirname(args.from_checkpoint)

    m.fit(
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        checkpoint_every=10,
        logdir=logdir
    )
