
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\Allen\Desktop\audio-conversion-network")

data_waveform, sample_rate = torchaudio.load(r"C:\Users\Allen\Desktop\discord\sfx\UwU - SOUND EFFECT.mp3", format="mp3")

print("This is the shape of the waveform: {}".format(data_waveform.size()))

print("This is the output for Sample rate of the waveform: {}".format(sample_rate))

plt.plot(data_waveform.t().numpy())
plt.savefig(r"C:\Users\Allen\Desktop\audio-conversion-network\waveform.png")

downsample_rate = sample_rate // 8

downsampler = torchaudio.transforms.Resample(
    sample_rate, downsample_rate, resampling_method='sinc_interpolation')

downsampled_waveform = downsampler(data_waveform)

print(torch.equal(downsampled_waveform[0], downsampled_waveform[1]))
test_data = downsampled_waveform[0]

torchaudio.save('test.wav', test_data.repeat(2, 1), downsample_rate)

pitch_shifter = torchaudio.transforms.PitchShift(
    downsample_rate, -4)

train_data = pitch_shifter(test_data)
torchaudio.save('train.wav', train_data.repeat(2, 1), downsample_rate)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(test_data.size()[0], test_data.size()[0]//2)
        self.fc2 = nn.Linear(test_data.size()[0]//2, test_data.size()[0])
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = Net()
print(net)

def train(model, train_data, test_data, epochs, learning_rate=0.001):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data.unsqueeze(0))
        loss = torch.sum(torch.abs(test_data - outputs))
        print(loss)
        loss.backward()
        optimizer.step()

train(net, train_data, test_data, 10)

prediction = net(train_data).detach()
torchaudio.save('temp.wav', prediction.unsqueeze(0).repeat(2, 1), downsample_rate)

plt.figure()
plt.plot(test_data.numpy())
plt.plot(train_data.numpy())
plt.savefig(r"C:\Users\Allen\Desktop\audio-conversion-network\train_test_waveform.png")
plt.figure()
plt.plot(prediction.numpy())
plt.savefig(r"C:\Users\Allen\Desktop\audio-conversion-network\prediction_waveform.png")