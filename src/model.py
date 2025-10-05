import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    '''
    A classic CNN for MNIST classification:
    Conv → ReLU → Conv → MaxPool → FC → FC → output (10 classes)
    Output used to : Predict labels, Generate softmax + entropy for state
    '''
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN(nn.Module):
    '''
    A simple MLP for value function approximation
    - Input = 11 (10 probs + 1 entropy)
    - Output = 11 actions (labels 0–9 + skip)
    '''
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
