### Audio samples 

Audio samples can be found [here](https://janvainer.github.io/speedyspeech/)

### Installation instructions
The code was tested with `python 3.7.3` and `cuda 10.0.130`.

```
git clone https://github.com/janvainer/speedyspeech.git
cd speedyspeech
git submodule init; git submodule update

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
To train the fertility model, durations of phonemes are needed.

1. Download the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) and unzip into `datasets/data/LJSpeech-1.1`
2. Train the duration extraction model
```
python code/duration_extractor.py -h  # display options
python code/duration_extractor.py --some_option value  # train the model, create logs for checkpoint
tensorboard --logdir=logs  # watch the training
```
3. Extract durations from the trained model - creates alignments.txt file in the LJSpeech-1.1 folder
```
python code/extract_durations.py logs/your_checkpoint code/datasets/data/LJSpeech-1.1
```
4. Train the fertility model
```
python code/speedyspeech.py -h  # display options
python code/speedyspeech.py  # train the model, create logs2 for checkpoint
tensorboard --logdir=logs2  # watch the training
```

### Inference
1. Download pretrained MelGAN checkpoint and set git head to the right commit
```
wget https://github.com/seungwonpark/melgan/releases/download/v0.1-alpha/nvidia_tacotron2_LJ11_epoch3200.pt \
    -O code/checkpoints/melgan_checkpoint.pt
cd code/melgan
git checkout 36d5071
```

2. Run inference
```
mkdir synthesized_audio
echo "One sentence. \nAnother sentence. | python code/inference.py <speedyspeech_checkpoint> <melgan_checkpoint> --audio_folder ~/synthesized_audio
cat text.txt | python code/inference.py <speedyspeech_checkpoint> <melgan_checkpoint> --device cuda
```
Files wil be added to the audio folder. The model does not handle numbers. please write everything in words.
The list of allowed symbols is specified in ```code/hparam.py```. 
