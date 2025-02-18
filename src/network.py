import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

def layer_init(layer, std=1.41, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SharedCNN(nn.Module):
    """Shared convolutional feature extractor for both Actor and Critic."""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=4, stride=2),  # 96x96x4 -> 47x47x8
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # 47x47x8 -> 23x23x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # 23x23x16 -> 11x11x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # 11x11x32 -> 5x5x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # 5x5x64 -> 3x3x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # 3x3x128 -> 1x1x256
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_layers(x).view(x.size(0), -1)  # Flatten for FC layers

class Actor(nn.Module):
    """Actor network for policy learning."""
    def __init__(self, shared_cnn, device):
        super().__init__()
        self.shared_cnn = shared_cnn  # Shared CNN feature extractor
        # Fully connected layers for policy
        self.fc1 = layer_init(nn.Linear(256, 100))
        # Five actions: turn left, turn right, accelerate, brake, do nothing
        self.fc2 = layer_init(nn.Linear(100, 5), std=0.01)

        self.to(device)

    def forward(self, x):
        x = self.shared_cnn(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return dist.Categorical(logits=logits)

class Critic(nn.Module):
    """Critic network for value estimation."""
    def __init__(self, shared_cnn, device):
        super().__init__()
        self.shared_cnn = shared_cnn  # Shared CNN feature extractor
        
        # Fully connected layers for value function (Critic)
        self.fc_value = layer_init(nn.Linear(256, 100))
        self.value_head = layer_init(nn.Linear(100, 1), std=1)

        self.to(device)

    def forward(self, x):
        features = self.shared_cnn(x)
        value = F.relu(self.fc_value(features))
        return self.value_head(value)  # Single scalar output