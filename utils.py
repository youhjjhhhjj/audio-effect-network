
import torch
import torchaudio

import os
from pathlib import Path
from typing import Iterator, Tuple


def load_wav(path: Path, resample_rate: int) -> torch.Tensor:
    """Load audio file and downsample it."""
    waveform, sr = torchaudio.load(path, format="wav")
    if waveform.shape[0] != 1:
        print("Audio file is stereo, proceeding as mono: " + str(path))
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != resample_rate:
        return torchaudio.functional.resample(waveform[0], orig_freq=sr, new_freq=resample_rate)
    return waveform[0]

def get_data_scale(data: torch.Tensor) -> Tuple[float, float]:
    """Return a scale and shift such that scale * data + shift results in a range between -1 and 1."""
    data_min = data.min()
    data_max = data.max()
    scale = 2 / (data_max - data_min)
    shift = -1 - (data_min * scale)
    return scale, shift

def list_files(path: Path) -> Iterator[Path]:
    """Generator for file paths in the directory."""
    for f in os.listdir(path.resolve()):
        if os.path.isfile(path/f):
            yield path/f
