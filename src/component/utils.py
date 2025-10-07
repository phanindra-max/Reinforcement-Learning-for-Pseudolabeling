import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import datasets, transforms
from collections import deque
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def get_ssl_mnist(labeled_size=100, unlabeled_size=2000, batch_size=16):
    '''
    - Loads MNIST
    - Shuffles + splits into:
        - Labeled subset
        - Unlabeled subset
    - Returns:
        - labeled_loader: used every training step
        - unlabeled_loader: passed into env
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    labeled_indices = indices[:labeled_size]
    unlabeled_indices = indices[labeled_size:labeled_size + unlabeled_size]

    labeled_set = Subset(full_dataset, labeled_indices)
    unlabeled_set = Subset(full_dataset, unlabeled_indices)

    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=1, shuffle=True)

    return labeled_loader, unlabeled_loader

class ReplayBuffer:
    '''
    - Stores (state, action, reward, next_state, done)
    - Supports sample() for batch training
    '''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

def retrain_downstream_model(model, pseudo_data, device="cpu", epochs=3, batch_size=16, lr=0.001):
    '''
    - Trains SimpleCNN on pseudo_labeled_data
    - Uses classic CrossEntropyLoss
    - Trains for a few epochs
    - Called at the end of each RL episode
    '''

    if not pseudo_data:
        print("No pseudo-labeled data to retrain on.")
        return model

    x_list, y_list = zip(*pseudo_data)
    x_tensor = torch.cat(x_list, dim=0)
    labels = [lbl for _, lbl in pseudo_data]
    print("Unique labels in pseudo-data:", set(labels))
    y_tensor = torch.tensor(y_list)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Retrain Epoch {epoch+1}/{epochs}, Loss={total_loss/len(loader):.4f}")

    return model

def evaluate_model(model, device):
    '''
    - Evaluates SimpleCNN on test set
    - Reports % accuracy
    - Logged per episode
    '''
    test_data = datasets.MNIST('../data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_data, batch_size=128)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    print(f"Test Accuracy: {correct / total:.2%}")

def evaluate_model_accuracy(model, device):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    test_data = datasets.MNIST('../data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_data, batch_size=128)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return correct / total  # return accuracy as float

def compute_reward(pred_label, true_label, action, probs):
    """Reward scaled by confidence."""
    if action == -1:
        return 0.0  # skip
    confidence = probs[action]
    if action == true_label:
        return confidence  # reward ∝ confidence
    else:
        return -0.5 * (1 - confidence)  # smaller penalty if low confidence

def get_mnist_subset(batch_size=1, size=1000):
    from torchvision import datasets, transforms
    from torch.utils.data import Subset

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    indices = list(range(size))
    subset = Subset(full_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader

class DownstreamModel(nn.Module):
    """A simple two-layer CNN for MNIST classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # The input x is initially (N, 784), so we must reshape it
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_downstream_model(self, initial_training=False, epochs=1):
        if initial_training:
            x_train, y_train = self.labeled_x, self.labeled_y
        else:
            if not self.newly_pseudo_labeled_data: return # Nothing to train on

            pseudo_x = torch.stack([item[0] for item in self.newly_pseudo_labeled_data])
            pseudo_y = torch.stack([item[1] for item in self.newly_pseudo_labeled_data])

            x_train = torch.cat([self.labeled_x, pseudo_x])
            y_train = torch.cat([self.labeled_y, pseudo_y])

        dataset = TensorDataset(x_train.view(-1, 1, 28, 28), y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.downstream_model.train()
        for _ in range(epochs):
            for x_b, y_b in loader:
                self.optimizer.zero_grad()
                out = self.downstream_model(x_b)
                loss = self.criterion(out, y_b.long())
                loss.backward()
                self.optimizer.step()

def _evaluate_model(self, x_data, y_data):
    self.downstream_model.eval()
    with torch.no_grad():
        # Reshape validation data for the CNN
        x_data = x_data.view(-1, 1, 28, 28)
        preds = torch.argmax(self.downstream_model(x_data), dim=1)
        return accuracy_score(y_data.cpu(), preds.cpu())
