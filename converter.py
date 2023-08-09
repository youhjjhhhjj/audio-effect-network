import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt
import os

from random import randint

os.chdir(r"C:\Users\Allen\Desktop\audio-conversion-network")

_sample_rate = 44100
_downsample_ratio = 8
_hidden_ratio = 2
_network_dim_s = 1  # number of seconds


downsample_rate = _sample_rate // _downsample_ratio
model_width = int(_network_dim_s * downsample_rate)

downsampler = torchaudio.transforms.Resample(
    _sample_rate, downsample_rate, resampling_method='sinc_interpolation')

def load_wav(path):
    data_waveform, sample_rate = torchaudio.load(path, format="wav")
    assert sample_rate == _sample_rate  # rate matches
    assert data_waveform.size()[0] == 1  # mono
    return downsampler(data_waveform)

# load train data
train_data = load_wav(r"C:\Users\Allen\Downloads\Swain_Original_Taunt_2.ogg")
torchaudio.save('train.wav', train_data, downsample_rate)

# load test data
test_data = load_wav(r"C:\Users\Allen\Desktop\audio-conversion-network\tmpvt4eexsy.wav")
torchaudio.save('test.wav', test_data, downsample_rate)
# test_data = torchaudio.transforms.PitchShift(downsample_rate, 4)(train_data).detach()
# torchaudio.save('test.wav', test_data, downsample_rate)

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

net = Net(model_width, _hidden_ratio)
print(net)
losses = train(net, train_data, test_data, 10)
# losses = train(net, train_data[:, :model_width], test_data[:, :model_width], 10)
plt.figure()
plt.plot(losses)
plt.savefig(r"C:\Users\Allen\Desktop\audio-conversion-network\losses.png")

# prediction = net(train_data).detach()
prediction = predict(net, train_data)
torchaudio.save('prediction.wav', prediction, downsample_rate)

plt.figure()
plt.plot(test_data.numpy())
plt.plot(train_data.numpy())
plt.savefig(r"C:\Users\Allen\Desktop\audio-conversion-network\train_test_waveform.png")
plt.figure()
plt.plot(prediction.numpy())
plt.savefig(r"C:\Users\Allen\Desktop\audio-conversion-network\prediction_waveform.png")