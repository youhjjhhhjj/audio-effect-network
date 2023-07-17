
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
model_width = _network_dim_s * downsample_rate

downsampler = torchaudio.transforms.Resample(
    _sample_rate, downsample_rate, resampling_method='sinc_interpolation')

def load_wav(path):
    data_waveform, sample_rate = torchaudio.load(path, format="wav")
    assert sample_rate == _sample_rate  # rate matches
    assert data_waveform.size()[0] == 1  # mono
    return downsampler(data_waveform)

# load train data
train_data = load_wav(r"C:\Users\Allen\Downloads\Swain Wav\taunts\Swain_Original_Taunt_2.ogg")
torchaudio.save('train.wav', train_data, downsample_rate)

# load test data
test_data = load_wav(r"C:\Users\Allen\Desktop\audio-conversion-network\tmpvt4eexsy.wav")
torchaudio.save('test.wav', test_data, downsample_rate)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(model_width, model_width // _hidden_ratio)
        self.fc2 = nn.Linear(model_width // _hidden_ratio, model_width)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = Net()
print(net)

def train(model, train_data, test_data, epochs, learning_rate=0.001):
    data_length = min(train_data.size()[1], test_data.size()[1])
    train_data = train_data[:, : data_length]
    test_data = test_data[:, : data_length]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(epochs):
        # create training sequence
        index = randint(0, data_length - model_width)
        train_seq = train_data[:, index : index + model_width]
        test_seq = test_data[:, index : index + model_width]
        # run backward pass
        optimizer.zero_grad()
        outputs = model(train_seq)
        loss = torch.sum(torch.abs(test_seq - outputs))
        print(loss)
        loss.backward()
        optimizer.step()

train(net, train_data, test_data, 10)

# prediction = net(train_data).detach()
prediction = net(train_data[:, 0 : model_width]).detach()
torchaudio.save('prediction.wav', prediction, downsample_rate)

plt.figure()
plt.plot(test_data.numpy())
plt.plot(train_data.numpy())
plt.savefig(r"C:\Users\Allen\Desktop\audio-conversion-network\train_test_waveform.png")
plt.figure()
plt.plot(prediction.numpy())
plt.savefig(r"C:\Users\Allen\Desktop\audio-conversion-network\prediction_waveform.png")