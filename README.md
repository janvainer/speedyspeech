# SpeedySpeech [[Paper link]](https://arxiv.org/pdf/2008.03802.pdf)

While recent neural sequence-to-sequence models have greatly improved the quality of speech synthesis, 
there has not been a system capable of 
fast training, fast inference and
high-quality audio synthesis at the same time. 
We propose a student-teacher network 
capable of high-quality faster-than-real-time spectrogram synthesis, with low requirements on computational resources and fast training time.
We show that self-attention layers are not necessary for generation of high quality audio. 
We utilize simple convolutional blocks with residual connections in both student and teacher networks and use only a single attention layer in the teacher model.
Coupled with a MelGAN vocoder, our model's voice quality was rated significantly higher than Tacotron2.
Our model can be efficiently trained on a single GPU and can run in real time even on a 
CPU.

Listen to our **audio samples [here](https://janvainer.github.io/speedyspeech/)**.

<a href="url"><img src="https://github.com/janvainer/speedyspeech/blob/master/img/speedyspeech.png" align="middle" height="360" ></a>


## Installation instructions
The code was tested with `python 3.7.3`, `cuda 10.0.130` and `GNU bash 5.0.3` on Ubuntu 19.04.

```
git clone https://github.com/janvainer/speedyspeech.git
cd speedyspeech

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Inference
**1. Download pretrained MelGAN** checkpoint
```
wget -O checkpoints/melgan.pth \
    https://github.com/seungwonpark/melgan/releases/download/v0.1-alpha/nvidia_tacotron2_LJ11_epoch3200.pt 
```

**2. Download pretrained SpeedySpeech** checkpoint from the latest release.
```
wget -O checkpoints/speedyspeech.pth \
    https://github.com/janvainer/speedyspeech/releases/download/v0.2/speedyspeech.pth 
```

**3. Run inference**
```
mkdir synthesized_audio
printf "One sentence. \nAnother sentence.\n" | python code/inference.py --audio_folder synthesized_audio
```
The model treats each line of input as an item in a batch.
To specify different checkpoints, what device to run on etc. use the following:
```
printf "One sentence. \nAnother sentence.\n" | python code/inference.py \
    --speedyspeech_checkpoint <speedyspeech_checkpoint> \
    --melgan_checkpoint <melgan_checkpoint> \
    --audio_folder synthesized_audio \
    --device cuda
```

Files wil be added to the audio folder. The model does not handle numbers. please write everything in words.
The list of allowed symbols is specified in ```code/hparam.py```. 

**4. Run inference server**
- Place SpeedySpeech and MelGAN checkpoints in the `checkpoints` folder.
```
checkpoints/
    melgan.pth
    speedyspeech.pth
```
And run the following commands. You should be able to open a simple webpage where you can
try to synthesize custom sentences.
```
cd code
python server/app.py  # go to http://127.0.0.1:5000/
python server/app.py --help

    usage: app.py [-h] [--speedyspeech_checkpoint SPEEDYSPEECH_CHECKPOINT]
        [--melgan_checkpoint MELGAN_CHECKPOINT] [--device DEVICE]

    optional arguments:
      -h, --help            show this help message and exit
      --speedyspeech_checkpoint SPEEDYSPEECH_CHECKPOINT
                            Checkpoint file for speedyspeech model
      --melgan_checkpoint MELGAN_CHECKPOINT
                            Checkpoint file for MelGan.
      --device DEVICE       What device to use.
```
<a href="url"><img src="https://github.com/janvainer/speedyspeech/blob/master/img/browser-inference.png" align="middle" height="180" ></a>

## Training
To train speedyspeech, durations of phonemes are needed.

**1. Download the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/)** and unzip into `datasets/data/LJSpeech-1.1`
```
wget -O code/datasets/data/LJSpeech-1.1.tar.bz2 \
    https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf code/datasets/data/LJSpeech-1.1.tar.bz2 -C code/datasets/data/
```
**2. Train the duration extraction model**
```
python code/duration_extractor.py -h  # display options
python code/duration_extractor.py \
    --some_option value
tensorboard --logdir=logs
```
**3. Extract durations from the trained model** - creates alignments.txt file in the LJSpeech-1.1 folder
```
python code/extract_durations.py logs/your_checkpoint code/datasets/data/LJSpeech-1.1 \
    --durations_filename my_durations.txt
```
**4. Train SpeedySpeech**
```
python code/speedyspeech.py -h
python code/speedyspeech.py \
    --durations_filename my_durations.txt
tensorboard --logdir=logs2
```
## License
This code is published under the BSD 3-Clause License.
1. `code/melgan` - [MelGAN](https://github.com/seungwonpark/melgan) by Seungwon Park (BSD 3-Clause License)
2. `code/utils/stft.py` - [torch-stft](https://github.com/pseeth/torch-stft) by Prem Seetharaman (BSD 3-Clause License)
3. `code/pytorch_ssim` - [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) by Po-Hsun-Su (MIT)
