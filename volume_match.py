import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\Allen\Desktop\audio-conversion-network")

data_waveform0, sample_rate0 = torchaudio.load(r"C:\Users\Allen\Downloads\Volibear Wav\Volibear_Original_R_0.ogg", format="wav")
data_waveform1, sample_rate1 = torchaudio.load(r"C:\Users\Allen\Downloads\Volibear Wav\converted\Volibear_Original_R_0.wav", format="wav")

min0 = data_waveform0.min()
min1 = data_waveform1.min()
range0 = data_waveform0.max() - min0
range1 = data_waveform1.max() - min1

scale = range0/range1
shift = min0 - (min1 * scale)

data_waveform2 = data_waveform1 * scale + shift

torchaudio.save('boosted.wav', data_waveform2, sample_rate1)




import soundfile as sf
import pyloudnorm as pyln

os.chdir(r"C:\Users\Allen\Desktop\audio-conversion-network\boost")

data_waveform0, sample_rate0 = sf.read(r"C:\Users\Allen\Downloads\Volibear Wav\Volibear_Original_R_0.ogg")
data_waveform1, sample_rate1 = sf.read(r"C:\Users\Allen\Downloads\Volibear Wav\converted\Volibear_Original_R_0.wav")
meter0 = pyln.Meter(sample_rate0)
loudness0 = meter0.integrated_loudness(data_waveform0)
sf.write('original.wav', data_waveform0, sample_rate0)
meter1 = pyln.Meter(sample_rate1)
loudness1 = meter1.integrated_loudness(data_waveform1)
sf.write('converted.wav', data_waveform1, sample_rate1)
data_waveform1_boosted = loudness_normalized_audio = pyln.normalize.loudness(data_waveform1, loudness1, loudness0)
sf.write('boosted.wav', data_waveform1_boosted, sample_rate1)