# Audio Classification Project

This project classifies audio samples using spectrogram images. The current milestone focuses on the dataset package and a runnable notebook that shows the model inputs and targets.

## Dataset

The repository includes a small example dataset under `data/example_data/` so the milestone notebook can run immediately after cloning the repo. The example dataset contains two spectrogram images per class:

- `Bass`
- `Hi Hat`
- `Kick`
- `Pad`
- `Snare`
- `Vocal`

Each example is a `.png` spectrogram. The data loader returns:

- input: a `torch.Tensor` image with shape `(3, 224, 224)`
- target: a `torch.long` integer label

## Install

From the repository root:

```bash
python -m pip install -e .
```

## Run the Notebook

Open and run:

```bash
notebooks/data_demo.ipynb
```

The notebook loads `data/example_data`, prints the class mapping, fetches one sample, creates a batch with `torch.utils.data.DataLoader`, and visualizes example spectrograms.

## Project Layout

```text
audio_classification/
  data/
    dataset.py       # PyTorch Dataset for spectrogram images
    dataloader.py    # DataLoader helper functions
data/
  example_data/      # small milestone dataset included in the repo
notebooks/
  data_demo.ipynb    # runnable milestone data demo
scripts/
  split.py           # dataset split utility
```
