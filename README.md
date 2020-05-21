## SpeedySpeech

While recent neural sequence-to-sequence models have greatly improved the quality of speech synthesis, 
there has not been a system capable of 
fast training, fast inference and
high-quality audio synthesis at the same time. 
We propose a student-teacher network 
capable of high-quality faster-than-real-time spectrogram synthesis, with low requirements on computational resources and fast training time.
We show that self-attention layers are not necessary for generation of high quality audio. 
We utilize simple convolutional blocks with residual connections in both student and teacher networks and use only a single attention layer in the teacher model.
Coupled with a MelGAN vocoder, our model's voice quality was rated significantly higher than Tacotron~2.
Our model can be efficiently trained on a single GPU and can run in real time even on a 
CPU.

See our audio samples [here](https://janvainer.github.io/speedyspeech/).

### Installation instructions
The code was tested with `python 3.7.3` and `cuda 10.0.130` on Ubuntu 19.04.

```
git clone https://github.com/janvainer/speedyspeech.git
cd speedyspeech
git submodule init; git submodule update

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Inference
1. Download pretrained MelGAN checkpoint and set git head to the right commit
```
wget https://github.com/seungwonpark/melgan/releases/download/v0.1-alpha/nvidia_tacotron2_LJ11_epoch3200.pt \
    -O checkpoints/melgan.pt
cd code/melgan
git checkout 36d5071
```

2. Download SpeedySpeech checkpoint from the latest release.
```
wget https://github.com/janvainer/speedyspeech/releases/download/v0.1/speedyspeech.pth \
    -O checkpoints/speedyspeech.pth
```

2. Run inference
```
mkdir synthesized_audio
echo "One sentence. \nAnother sentence. | python code/inference.py --audio_folder synthesized_audio
```
The model treats each line of input as an item in a batch.
To specify different checkpoints, what device to run on etc. use the following:
```
echo "One sentence. \nAnother sentence. | python code/inference.py \
    --speedyspeech_checkpoint <speedyspeech_checkpoint> \
    --melgan_checkpoint <melgan_checkpoint> \
    --audio_folder synthesized_audio \
    --device cuda
```

Files wil be added to the audio folder. The model does not handle numbers. please write everything in words.
The list of allowed symbols is specified in ```code/hparam.py```. 

### Training
To train the fertility model, durations of phonemes are needed.

1. Download the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) and unzip into `datasets/data/LJSpeech-1.1`
2. Train the duration extraction model
```
python code/duration_extractor.py -h  # display options
python code/duration_extractor.py \
    --some_option value
tensorboard --logdir=logs
```
3. Extract durations from the trained model - creates alignments.txt file in the LJSpeech-1.1 folder
```
python code/extract_durations.py logs/your_checkpoint code/datasets/data/LJSpeech-1.1
```
4. Train the fertility model
```
python code/speedyspeech.py -h
python code/speedyspeech.py
    --some_option some_value
tensorboard --logdir=logs2
```
