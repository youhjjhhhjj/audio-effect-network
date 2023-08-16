
import torch.nn as nn

class Feedforward(nn.Module):
    def __init__(self, model_width, hidden_ratio):
        super(Feedforward, self).__init__()
        hidden_dim = int(model_width * hidden_ratio)
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