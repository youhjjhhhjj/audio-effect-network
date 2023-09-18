
from models import *
from utils import load_wav

import torch
import torchaudio

import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path
from random import randrange
from typing import List


loss_function = torch.nn.CosineSimilarity(dim=1)

def train(model: torch.nn.Module, input_dataset: List[torch.Tensor], target_dataset: List[torch.Tensor], epochs: int, batch_size: int = 8, learning_rate: float = 0.0001) -> List[float]:
    """Train the model with Adam optimization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    # training loop
    for i in range(epochs):
        input_batch = []
        target_batch = []
        for j in range(batch_size):
            input_data = input_dataset[j % len(input_dataset)]
            target_data = target_dataset[j % len(input_dataset)]
            # create training sequence
            data_length = min(input_data.shape[1], target_data.shape[1])
            index = randrange(0, data_length)
            input_batch.append(input_data[:, index])
            target_batch.append(target_data[:, index])
        # run backward pass
        optimizer.zero_grad()
        outputs = model(torch.stack(input_batch, dim=0))
        loss = 1 - loss_function(outputs, torch.stack(target_batch, dim=0)).mean()
        print(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs=2)
    parser.add_argument("-epochs", help="Number of training epochs", type=int, default=5)
    parser.add_argument("-batch_size", "-bs", help="Size of training batch", type=int, default=8)
    parser.add_argument("-hidden_ratio", "-hr", help="Ratio of hidden dimension to input dimension", type=float, default=1.0)
    parser.add_argument("-n_fft", help="Number of fft bins for spectrogram", type=int, default=512)
    parser.add_argument("-downsample_rate", "-dr", help="Frequency to downsample inputs for faster running", type=float, default=11025)
    parser.add_argument("-save_model", "-sm", help="Filename if the model is to be saved", type=str, default="")
    parser.add_argument("-load_model", "-lm", help="Filename if the model is to be loaded (warm start)", type=str, default="")

    args=parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    hidden_ratio = args.hidden_ratio
    n_fft = args.n_fft
    downsample_rate = args.downsample_rate
    save_model_name = args.save_model
    load_model_name = args.load_model
    input_dir = Path(sys.argv[1]).resolve()
    target_dir = Path(sys.argv[2]).resolve()

    model_dim = n_fft // 2 + 1 

    input_dataset = []
    target_dataset = []

    for input_file, target_file in zip(sorted(os.listdir(input_dir)), sorted(os.listdir(target_dir))):
        input_waveform = load_wav(input_dir/input_file, downsample_rate)
        target_waveform = load_wav(target_dir/target_file, downsample_rate)
        input_dataset.append(torch.stft(input_waveform, n_fft, return_complex=True).abs())
        target_dataset.append(torch.stft(target_waveform, n_fft, return_complex=True).abs())
    
    net = Feedforward(model_dim, hidden_ratio)
    if load_model_name != "":
        params = torch.load((Path().resolve() / "weights" / load_model_name).with_suffix(".pt"))
        net.load_state_dict(params)
        print("Successfully loaded " + load_model_name + ".pt")
    print(net)
    print(sum(p.numel() for p in net.parameters()))
    losses = train(net, input_dataset, target_dataset, epochs, batch_size)
    if save_model_name != "":
        torch.save(net.state_dict(), (Path().resolve() / "weights" / save_model_name).with_suffix(".pt"))
        print("Successfully saved " + save_model_name + ".pt")
    plt.figure()
    plt.plot(losses)
    plt.savefig("losses.png")
