import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt
import sys

from random import randint

def load_wav(path):
    data_waveform, sample_rate = torchaudio.load(path, format="wav")
    assert sample_rate == _sample_rate  # rate matches
    assert data_waveform.size()[0] == 1  # mono
    return downsampler(data_waveform)

class Net(nn.Module):
    def __init__(self, model_width, hidden_ratio):
        super(Net, self).__init__()
        hidden_dim = int(model_width // hidden_ratio)
        self.fc1 = nn.Linear(model_width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, model_width)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def train(model, train_data, test_data, epochs, learning_rate=0.001):
    data_length = min(train_data.size()[1], test_data.size()[1])
    train_data = train_data[:, : data_length]
    test_data = test_data[:, : data_length]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for i in range(epochs):
        # create training sequence
        index = randint(0, data_length - model_width)
        train_seq = train_data[:, index : index + model_width]
        test_seq = test_data[:, index : index + model_width]
        scale, shift = get_data_scale(train_seq)
        train_seq = train_seq * scale + shift
        test_seq = test_seq * scale + shift
        # run backward pass
        optimizer.zero_grad()
        outputs = model(train_seq)
        # loss_function = nn.MSELoss()
        loss = (test_seq - outputs).square().sum()
        # loss = loss_function(outputs, test_seq)
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
    epochs = int(sys.argv[3])
    hidden_ratio = float(sys.argv[4])
    network_dim_s = float(sys.argv[5])  # number of seconds
    downsample_ratio = int(sys.argv[6])

    _sample_rate = 44100

    downsample_rate = _sample_rate // downsample_ratio
    model_width = int(network_dim_s * downsample_rate)

    downsampler = torchaudio.transforms.Resample(
        _sample_rate, downsample_rate, resampling_method='sinc_interpolation')

    train_data = load_wav(sys.argv[1])
    test_data = load_wav(sys.argv[2])    
    
    net = Net(model_width, hidden_ratio)
    print(net)
    losses = train(net, train_data, test_data, epochs)
    # losses = train(net, train_data[:, :model_width], test_data[:, :model_width], 10)
    plt.figure()
    plt.plot(losses)
    plt.savefig("losses.png")

    prediction = predict(net, train_data)
    torchaudio.save('prediction.wav', prediction, downsample_rate)
