
from models import Feedforward
from utils import load_wav, list_files

import torch
import torchaudio

import sys
import argparse
from pathlib import Path
from typing import List


def predict(model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
    """Generate one prediction."""
    predictions = []
    for i in range(data.shape[1]):
        with torch.no_grad():
            data_seq = data[:, i]
            prediction = model(data_seq).detach()
            predictions.append(prediction.unsqueeze(1))
    return torch.cat(predictions, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs=2)
    parser.add_argument("-hidden_ratio", "-hr", help="Ratio of hidden dimension to input dimension", type=float, default=1.0)
    parser.add_argument("-n_fft", help="Number of fft bins for spectrogram", type=int, default=512)
    parser.add_argument("-downsample_rate", "-dr", help="Frequency to downsample inputs for faster running", type=float, default=11025)
    parser.add_argument("-load_model", "-lm", help="Filename if the model is to be loaded (warm start)", type=str)

    args=parser.parse_args()
    hidden_ratio = args.hidden_ratio
    n_fft = args.n_fft
    downsample_rate = args.downsample_rate
    load_model_name = args.load_model
    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    model_dim = n_fft // 2 + 1 

    net = Feedforward(model_dim, hidden_ratio)
    params = torch.load((Path().resolve() / "weights" / load_model_name).with_suffix(".pt"))
    net.load_state_dict(params)
    print("Successfully loaded " + load_model_name + ".pt")

    for f in list_files(input_dir):        
        input_waveform = load_wav(f, downsample_rate)
        input_data = torch.stft(input_waveform, n_fft, return_complex=True)
        prediction = predict(net, input_data.abs())
        prediction_waveform = torch.istft(torch.polar(prediction, input_data.angle().squeeze(dim=0)), n_fft, return_complex=False)
        torchaudio.save(output_dir / Path(f).with_suffix(".wav").name, prediction_waveform.unsqueeze(0), downsample_rate)
        print("Successfully processed " + f.name)
