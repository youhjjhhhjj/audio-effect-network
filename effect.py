from models import Feedforward
import torch
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

from random import randint

def load_wav(path, sample_rate):
    # TODO resample if necessary
    wav = torchaudio.load(path, format="wav")
    assert wav[1] == sample_rate  # rate matches
    assert wav[0].size()[0] == 1  # mono
    return downsampler(wav[0])

def train(model, input_data, target_data, epochs, learning_rate=0.001):
    data_length = min(input_data.size()[1], target_data.size()[1])
    input_data = input_data[:, : data_length]
    target_data = target_data[:, : data_length]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for i in range(epochs):
        # create training sequence
        index = randint(0, data_length - model_width)
        input_seq = input_data[:, index : index + model_width]
        target_seq = target_data[:, index : index + model_width]
        scale, shift = get_data_scale(input_seq)
        input_seq = input_seq * scale + shift
        target_seq = target_seq * scale + shift
        # run backward pass
        optimizer.zero_grad()
        outputs = model(input_seq)
        # loss_function = nn.MSELoss()
        loss = (target_seq - outputs).square().sum()
        # loss = loss_function(outputs, target_seq)
        print(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def predict(model, data):
    predictions = []
    for i in range(0, data.numel(), model_width):
        with torch.no_grad():
            data_seq = data[:, i : i + model_width]
            scale, shift = get_data_scale(data_seq)
            data_seq = data_seq * scale + shift
            prediction = model(F.pad(data_seq, (0, model_width - data_seq.numel()))).detach()
            prediction = (prediction - shift) / scale
            predictions.append(prediction[:, :data_seq.numel()])
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
    parser.add_argument("-network_dim", "-nd", help="How many seconds of audio for the network to process at once", type=float, default=1)
    parser.add_argument("-downsample_ratio", "-dr", help="Factor by which to downsample inputs for faster running", type=float, default=4)
    parser.add_argument("-sample_rate", "-sr", "-hz", help="Frequency of input audio", type=int, default=44100)
    parser.add_argument("-save_model", "-sm", help="Filename if the model is to be saved", type=str, default="")

    args=parser.parse_args()
    epochs = args.epochs
    hidden_ratio = args.hidden_ratio
    network_dim = args.network_dim
    downsample_ratio = args.downsample_ratio
    sample_rate = args.sample_rate
    save_model_name = args.save_model

    downsample_rate = int(sample_rate / downsample_ratio)
    model_width = int(network_dim * downsample_rate)

    downsampler = torchaudio.transforms.Resample(
        sample_rate, downsample_rate, resampling_method='sinc_interpolation')

    input_data = load_wav(sys.argv[1], sample_rate)
    target_data = load_wav(sys.argv[2], sample_rate)    
    
    net = Feedforward(model_width, hidden_ratio)
    print(net)
    losses = train(net, input_data, target_data, epochs)
    if save_model_name != "":
        torch.save(net.state_dict(), Path().absolute() / "weights" / (save_model_name + ".pth"))
    plt.figure()
    plt.plot(losses)
    plt.savefig("losses.png")

    prediction = predict(net, target_data)
    torchaudio.save('prediction.wav', prediction, downsample_rate)
