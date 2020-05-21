## Code organization


```
code
├── duration_extractor.py         # the TEACHER MODEL for duration extraction
├── speedyspeech.py               # The STUDENT MODEL for spectrogram synthesis
├── extract_durations.py          # extract durations from audio with teacher checkpoint
├── functional.py                 # masks, padding, attention
├── get_dataset_stats.py          # extract dataset stats for normalization
├── hparam.py                     # hyperparameters
├── inference.py                  # synthesize audio with pretrained checkpoint
├── layers.py                     # Pytorch modules
├── losses.py                     # Masked pytorch losses
├── melgan/                       # pulled from https://github.com/seungwonpark/melgan
├── pytorch_ssim/                 # pulled from https://github.com/Po-Hsun-Su/pytorch-ssim
├── stft.py                       # On-the fly mel spectrogram batch calculation
├── datasets
│   └── AudioDataset.py           # data loading
└── utils
    ├── augment.py                # spectrogram augmentations for teacher model
    ├── dynamic_time_warping.py   # dtw alignment plotting
    ├── masked.py                 # masking utils (masked mean etc)
    ├── optim.py                  # schedulers
    ├── text.py                   # Text processing, phoneme conversion
    ├── torch_stft.py             # pulled from https://github.com/pseeth/torch-stft
    └── transform.py              # normalization utils
```
