### Audio samples 

Audio samples can be found [here](https://janvainer.github.io/efficient-neural-speech-synthesis/)

### Installation instructions
The code was tested with `python 3.7.3` and `cuda 10.0.130`.

```
git clone https://github.com/LordOfLuck/convolutional_tts.git
cd convolutional_tts
git submodule init; git submodule update

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
To train the fertility model, durations of phonemes are needed.

1. Download the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) and unzip into `datasets/data/LJSpeech-1.1`
2. Train the duration prediction model
```
python code/conv_tacotron.py -h  # display options
python code/conv_tacotron.py --some_option value  # train the model, create logs for checkpoint
tensorboard --logdir=logs  # watch the training
```
3. Extract durations from the trained model - creates alignments.txt file in the LJSpeech-1.1 folder
```
python code/extract_durations.py logs/your_checkpoint code/datasets/data/LJSpeech-1.1
```
4. Train the fertility model
```
python code/fertility_model.py -h  # display options
python code/fertility_model.py  # train the model, create logs2 for checkpoint
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
echo "One sentence. \nAnother sentence. | python code/inference.py checkpoint1 checkpoint2 --audio_folder ~/audio
cat text.txt | python code/inference.py checkpoint1 checkpoint2 --device cuda
```
Files wil be added to the audio folder. The model does not handle numbers. please write everything in words.
The list of allowed symbols is specified in ```code/hparam.py```. 

Measured inference speed for the sentence *"If you want to build a ship, 
don't drum up people to collect wood and don't assign them tasks and work, 
but rather teach them to long for the endless immensity of the sea."*
The time is in seconds.

#####GPU
|batch          |spectrogram    |audio          |total          |
|---------------|---------------|---------------|---------------|
|1              |0.032          |0.165          |0.197          |
|2              |0.035          |0.325          |0.359          |
|4              |0.05           |0.647          |0.697          |
|8              |0.097          |1.291          |1.388          |
|16             |0.203          |4.065          |4.268          |

#####CPU
|batch          |spectrogram    |audio          |total          |
|---------------|---------------|---------------|---------------|
|1              |0.105          |1.702          |1.808          |
|2              |0.137          |3.211          |3.348          |
|4              |0.263          |6.788          |7.051          |
|8              |0.591          |14.061         |14.652         |

### MOS

Total respondents:  27.0

|model                    |MOS                      |95 % CI                  |
|-------------------------|-------------------------|-------------------------|
|fertility                |75.0                     |(-2.04, 2.04)            |
|tacotron_2               |62.36                    |(-2.4, 2.26)             |
|fertiliity_griffin       |46.45                    |(-2.57, 2.43)            |
|reference                |97.98                    |(-0.83, 0.7)             |
|deep_voice               |42.43                    |(-2.38, 2.56)            |
