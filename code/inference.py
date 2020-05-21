"""Synthesize audio from text

echo "One sentence. \nAnother sentence. | python code/inference.py checkpoint1 checkpoint2 --device cuda --audio_folder ~/audio
cat text.txt | python code/inference.py checkpoint1 checkpoint2 --device cuda

Run from the project root.
Audios are by default saved to ~/audio.
Does not handle numbers - write everything in words.

usage: inference.py [-h] [--speedyspeech_checkpoint SPEEDYSPEECH_CHECKPOINT]
                    [--melgan_checkpoint MELGAN_CHECKPOINT] [--device DEVICE]
                    [--audio_folder AUDIO_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --speedyspeech_checkpoint SPEEDYSPEECH_CHECKPOINT
                        Checkpoint file for speedyspeech model
  --melgan_checkpoint MELGAN_CHECKPOINT
                        Checkpoint file for MelGan.
  --device DEVICE       What device to use.
  --audio_folder AUDIO_FOLDER
                        Where to save audios
"""
import argparse, sys, os, time
import torch
from librosa.output import write_wav

from speedyspeech import SpeedySpeech
from melgan.model.generator import Generator
from melgan.utils.hparams import HParam
from hparam import HPStft, HPText
from utils.text import TextProcessor
from functional import mask

parser = argparse.ArgumentParser()
parser.add_argument("--speedyspeech_checkpoint", default='checkpoints/speedyspeech.pth', type=str, help="Checkpoint file for speedyspeech model")
parser.add_argument("--melgan_checkpoint", default='checkpoints/melgan.pth', type=str, help="Checkpoint file for MelGan.")
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',  help="What device to use.")
parser.add_argument("--audio_folder", type=str, default="synthesized_audio", help="Where to save audios")
args = parser.parse_args()

print('Loading model checkpoints')
m = SpeedySpeech(
    device=args.device
).load(args.speedyspeech_checkpoint, map_location=args.device)
m.eval()

checkpoint = torch.load(args.melgan_checkpoint, map_location=args.device)
hp = HParam("code/melgan/config/default.yaml")
melgan = Generator(hp.audio.n_mel_channels).to(args.device)
melgan.load_state_dict(checkpoint["model_g"])
melgan.eval(inference=False)

print('Processing text')
txt_processor = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
text = [t.strip() for t in sys.stdin.readlines()]

phonemes, plen = txt_processor(text)
# append more zeros - avoid cutoff at the end of the largest sequence
phonemes = torch.cat((phonemes, torch.zeros(len(phonemes), 5).long() ), dim=-1)
phonemes = phonemes.to(args.device)

print('Synthesizing')
# generate spectrograms
with torch.no_grad():
    spec, durations = m((phonemes, plen))


# invert to log(mel-spectrogram)
spec = m.collate.norm.inverse(spec)

# mask with pad value expected by MelGan
msk = mask(spec.shape, durations.sum(dim=-1).long(), dim=1).to(args.device)
spec = spec.masked_fill(~msk, -11.5129)

# Append more pad frames to improve end of the longest sequence
spec = torch.cat((spec.transpose(2,1), -11.5129*torch.ones(len(spec), HPStft.n_mel, 5).to(args.device)), dim=-1)

# generate audio
with torch.no_grad():
    audio = melgan(spec).squeeze(1)

print('Saving audio')
# TODO: cut audios to proper length
for i,a in enumerate(audio.detach().cpu().numpy()):
    write_wav(os.path.join(args.audio_folder,f'{i}.wav'), a, HPStft.sample_rate, norm=False)
