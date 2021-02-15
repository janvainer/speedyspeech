import sys, os, argparse
from io import BytesIO

import torch
import numpy as np
from scipy.io.wavfile import write
from flask import Flask, render_template, request, make_response

# insert python path to allow imports from parent dirs
#sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())

# SpeedySpeech imports
from hparam import HPStft, HPText
from utils.text import TextProcessor
from functional import mask

from speedyspeech import SpeedySpeech
from melgan.model.generator import Generator
from melgan.utils.hparams import HParam
from functional import mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speedyspeech_checkpoint",
                        default='../checkpoints/speedyspeech.pth',
                        type=str, help="Checkpoint file for speedyspeech model")
    parser.add_argument("--melgan_checkpoint",
                        default='../checkpoints/melgan.pth',
                        type=str, help="Checkpoint file for MelGan.")
    parser.add_argument("--device",
                        type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="What device to use.")
    args = parser.parse_args()
    return args


class SpeedySpeechInference:
    def __init__(self, speedyspeech_checkpoint, melgan_checkpoint, device):
        self.device = device
        self.speedyspeech = self._setup_speedyspeech(speedyspeech_checkpoint)
        self.melgan = self._setup_melgan(melgan_checkpoint)
        self.txt_processor = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
        self.bit_depth = 16
        self.sample_rate = 22050

    def _setup_speedyspeech(self, checkpoint):
        speedyspeech = SpeedySpeech(
            device=self.device
        ).load(checkpoint, map_location=self.device)
        speedyspeech.eval()
        return speedyspeech

    def _setup_melgan(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        hp = HParam("../code/melgan/config/default.yaml")
        melgan = Generator(hp.audio.n_mel_channels).to(self.device)
        melgan.load_state_dict(checkpoint["model_g"])
        melgan.eval(inference=False)
        return melgan

    def synthesize(self, input_text):
        print(input_text)
        text = [input_text.strip()]
        phonemes, plen = self.txt_processor(text)

        # append more zeros - avoid cutoff at the end of the largest sequence
        phonemes = torch.cat((phonemes, torch.zeros(len(phonemes), 5).long() ), dim=-1)
        phonemes = phonemes.to(self.device)

        # generate spectrograms
        with torch.no_grad():
            spec, durations = self.speedyspeech((phonemes, plen))

        # invert to log(mel-spectrogram)
        spec = self.speedyspeech.collate.norm.inverse(spec)

        # mask with pad value expected by MelGan
        msk = mask(spec.shape, durations.sum(dim=-1).long(), dim=1).to(self.device)
        spec = spec.masked_fill(~msk, -11.5129)

        # Append more pad frames to improve end of the longest sequence
        spec = torch.cat((
            spec.transpose(2,1),
            -11.5129 * torch.ones(len(spec), HPStft.n_mel, 5).to(self.device)
        ), dim=-1)

        # generate audio
        with torch.no_grad():
            audio = self.melgan(spec).squeeze(1)
            audio = audio.detach().cpu().numpy()[0]

        # denormalize
        x = 2 ** self.bit_depth - 1
        audio = np.int16(audio * x)
        return audio


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

args = get_args()
speedyspeech = SpeedySpeechInference(
    args.speedyspeech_checkpoint,
    args.melgan_checkpoint,
    args.device
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/synt/<text>',methods=['GET'])
def synt(text):
    buf = BytesIO()
    waveform_integers = speedyspeech.synthesize(text)
    write(buf, speedyspeech.sample_rate, waveform_integers)
    response = make_response(buf.getvalue())
    buf.close()
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
    return response;


app.run(debug=True)
