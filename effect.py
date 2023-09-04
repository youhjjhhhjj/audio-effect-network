from models import Feedforward
import torch
import torchaudio

import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path

from random import randrange


def load_wav(path, sample_rate):
    # TODO resample if necessary
    waveform, sr = torchaudio.load(path, format="wav")
    assert sr == sample_rate  # rate matches
    if waveform.shape[0] != 1:
        print("Audio file is stereo, proceeding as mono: " + str(path))
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return downsampler(waveform[0])

cossim = torch.nn.CosineSimilarity(dim=1)

def train(model, input_dataset, target_dataset, epochs, batch_size=8, learning_rate=0.0001):
    dataset_length = len(input_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for i in range(epochs):
        input_batch = []
        target_batch = []
        for j in range(batch_size):
            input_data = input_dataset[j % dataset_length]
            target_data = target_dataset[j % dataset_length]
            # create training sequence
            data_length = min(input_data.shape[1], target_data.shape[1])
            index = randrange(0, data_length)
            input_batch.append(input_data[:, index])
            target_batch.append(target_data[:, index])
        # run backward pass
        optimizer.zero_grad()
        outputs = model(torch.stack(input_batch, dim=0))
        loss = 1 - cossim(outputs, torch.stack(target_batch, dim=0)).mean()
        print(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def predict(model, data):
    predictions = []
    for i in range(data.shape[1]):
        with torch.no_grad():
            data_seq = data[:, i]
            prediction = model(data_seq).detach()
            predictions.append(prediction.unsqueeze(1))
    return torch.cat(predictions, dim=1)

def get_data_scale(data):
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    scale = 2/data_range
    shift = -1 - (data_min * scale)
    return scale, shift

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files",nargs=2)
    parser.add_argument("-epochs", help="Number of training epochs", type=int, default=5)
    parser.add_argument("-batch_size", "-bs", help="Size of training batch", type=int, default=8)
    parser.add_argument("-hidden_ratio", "-hr", help="Ratio of hidden dimension to input dimension", type=float, default=1.0)
    parser.add_argument("-n_fft", help="Number of fft bins for spectrogram", type=int, default=512)
    parser.add_argument("-downsample_ratio", "-dr", help="Factor by which to downsample inputs for faster running", type=float, default=4)
    parser.add_argument("-sample_rate", "-sr", "-hz", help="Frequency of input audio", type=int, default=44100)
    parser.add_argument("-save_model", "-sm", help="Filename if the model is to be saved", type=str, default="")

    args=parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    hidden_ratio = args.hidden_ratio
    n_fft = args.n_fft
    downsample_ratio = args.downsample_ratio
    sample_rate = args.sample_rate
    save_model_name = args.save_model
    input_dir = Path(sys.argv[1]).resolve()
    target_dir = Path(sys.argv[2]).resolve()

    downsample_rate = int(sample_rate / downsample_ratio)
    model_dim = n_fft // 2 + 1
    
    downsampler = torchaudio.transforms.Resample(sample_rate, downsample_rate, resampling_method='sinc_interpolation')    

    input_dataset = []
    target_dataset = []

    for input_file, target_file in zip(sorted(os.listdir(sys.argv[1])), sorted(os.listdir(sys.argv[2]))):
        input_waveform = load_wav(input_dir/input_file, sample_rate)
        target_waveform = load_wav(target_dir/target_file, sample_rate)
        input_dataset.append(torch.stft(input_waveform, n_fft, return_complex=True).abs())
        target_dataset.append(torch.stft(target_waveform, n_fft, return_complex=True).abs())
    
    net = Feedforward(model_dim, hidden_ratio)
    print(net)
    print(sum(p.numel() for p in net.parameters()))
    losses = train(net, input_dataset, target_dataset, epochs, batch_size)
    if save_model_name != "":
        torch.save(net.state_dict(), Path().absolute() / "weights" / (save_model_name + ".pth"))
    plt.figure()
    plt.plot(losses)
    plt.savefig("losses.png")

    # prediction = predict(net, input_data)
    # prediction_audio = torch.istft(torch.polar(prediction, input_spectrum.angle().squeeze(dim=0)), n_fft, return_complex=False)
    # torchaudio.save('prediction.wav', prediction_audio.unsqueeze(0), downsample_rate)
