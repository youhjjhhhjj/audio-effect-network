
import torch.nn as nn

class Convolutional(nn.Module):
    def __init__(self, model_width, hidden_ratio):
        super(Convolutional, self).__init__()
        hidden_dim = int(model_width * hidden_ratio)
        self.conv1 = nn.Conv1d(model_width, hidden_dim, kernel_size=5, padding=2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.act2 = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, model_width)
        
    def forward(self, x):
        x = self.conv1(x.unsqueeze(dim=2))
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.fc(x.squeeze(dim=2))
        return x