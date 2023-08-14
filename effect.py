from models import Feedforward
import torch
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

from random import randrange


def load_wav(path, sample_rate):
    # TODO resample if necessary
    wav = torchaudio.load(path, format="wav")
    assert wav[1] == sample_rate  # rate matches
    if wav[0].shape[0] != 1:
        print("Audio file is stereo, proceeding as mono: " + path)
    return downsampler(wav[0][0])

cossim = torch.nn.CosineSimilarity(dim=0)

def train(model, input_data, target_data, epochs, learning_rate=0.0001):
    data_length = min(input_data.shape[1], target_data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for i in range(epochs):
        # create training sequence
        index = randrange(0, data_length)
        input_seq = input_data[:, index]
        target_seq = target_data[:, index]
        # run backward pass
        optimizer.zero_grad()
        outputs = model(input_seq)
        loss = 1 - cossim(outputs, target_seq).sum()
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
    parser.add_argument("-hidden_ratio", "-hr", help="Ratio of hidden dimension to input dimension", type=float, default=1.0)
    parser.add_argument("-n_fft", help="Number of fft bins for spectrogram", type=int, default=512)
    parser.add_argument("-downsample_ratio", "-dr", help="Factor by which to downsample inputs for faster running", type=float, default=4)
    parser.add_argument("-sample_rate", "-sr", "-hz", help="Frequency of input audio", type=int, default=44100)
    parser.add_argument("-save_model", "-sm", help="Filename if the model is to be saved", type=str, default="")

    args=parser.parse_args()
    epochs = args.epochs
    hidden_ratio = args.hidden_ratio
    n_fft = args.n_fft
    downsample_ratio = args.downsample_ratio
    sample_rate = args.sample_rate
    save_model_name = args.save_model

    downsample_rate = int(sample_rate / downsample_ratio)
    model_dim = n_fft // 2 + 1
    
    downsampler = torchaudio.transforms.Resample(sample_rate, downsample_rate, resampling_method='sinc_interpolation')    

    input_audio = load_wav(sys.argv[1], sample_rate)
    target_audio = load_wav(sys.argv[2], sample_rate)    

    input_spectrum = torch.stft(input_audio, n_fft, return_complex=True)
    input_data = input_spectrum.abs().squeeze(dim=0)
    target_data = torch.stft(target_audio, n_fft, return_complex=True).abs().squeeze(dim=0)
    
    net = Feedforward(model_dim, hidden_ratio)
    print(net)
    print(sum(p.numel() for p in net.parameters()))
    losses = train(net, input_data, target_data, epochs)
    if save_model_name != "":
        torch.save(net.state_dict(), Path().absolute() / "weights" / (save_model_name + ".pth"))
    plt.figure()
    plt.plot(losses)
    plt.savefig("losses.png")

    prediction = predict(net, input_data)
    prediction_audio = torch.istft(torch.polar(prediction, input_spectrum.angle().squeeze(dim=0)), n_fft, return_complex=False)
    torchaudio.save('prediction.wav', prediction_audio.unsqueeze(0), downsample_rate)
