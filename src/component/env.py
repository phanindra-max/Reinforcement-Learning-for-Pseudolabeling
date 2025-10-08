import gymnasium as gym
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.nn.functional import softmax
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Main Environment Class
# -----------------------------
class PseudoLabelEnv(gym.Env):

    def __init__(self, NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION):
            super().__init__()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            labeled_data, unlabeled_data, validation_data = self._load_data(NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION)

            self.labeled_x, self.labeled_y, self.unlabeled_x, self.unlabeled_y_true = \
                labeled_data[0].to(self.device), labeled_data[1].to(self.device), \
                unlabeled_data[0].to(self.device), unlabeled_data[1].to(self.device)
            self.val_x, self.val_y = validation_data[0].to(self.device), validation_data[1].to(self.device)

            self.num_classes = 10
            self.feature_dim = self.labeled_x.shape[1]
            state_dim = self.feature_dim + self.num_classes + 1
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
            self.action_space = gym.spaces.Discrete(self.num_classes + 1)
            
            # Downstream Model
            self.downstream_model = DownstreamModel(self.device)
            self.downstream_model.train_model((self.labeled_x, self.labeled_y), [])

    # -----------------------------
    # Load Data
    # -----------------------------
    def _load_data(self, NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION):
            transform = transforms.ToTensor()
            mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

            all_data = mnist_train.data.float() / 255.0
            all_targets = mnist_train.targets

            data_temp, val_data, targets_temp, val_targets = train_test_split(
                all_data, all_targets, test_size=NUM_VALIDATION, stratify=all_targets, random_state=42
            )

            labeled_x, unlabeled_x, labeled_y, unlabeled_y_true = train_test_split(
                data_temp, targets_temp, train_size=NUM_LABELED, test_size=NUM_UNLABELED, stratify=targets_temp, random_state=42
            )

            labeled_data = (labeled_x.view(-1, 784), labeled_y)
            unlabeled_data = (unlabeled_x.view(-1, 784), unlabeled_y_true)
            validation_data = (val_data.view(-1, 784), val_targets)

            return labeled_data, unlabeled_data, validation_data
    
    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        self.newly_pseudo_labeled_data = []
        self.episode_reward = 0.0

        perm = torch.randperm(self.unlabeled_x.size(0))
        self.unlabeled_x = self.unlabeled_x[perm]
        self.unlabeled_y_true = self.unlabeled_y_true[perm]
        
        self.baseline_accuracy = self.downstream_model.evaluate_model(self.val_x, self.val_y)

        return self._get_state(), {}

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):
        img_idx = self.current_idx
        reward = 0
        
        if action < self.num_classes:
            true_label = self.unlabeled_y_true[img_idx]
            preds = self.downstream_model.get_predictions(self.unlabeled_x[img_idx].unsqueeze(0))
            pred_label = torch.argmax(preds).item()
            reward = self.calculate_reward(pred_label, true_label, action, preds)
            self.episode_reward += reward
            
            self.newly_pseudo_labeled_data.append(
                (self.unlabeled_x[img_idx], torch.tensor(action, device=self.device))
            )

        else:  # Action is skip (action == num_classes)
            reward = self.calculate_reward(None, None, action, None)

        self.current_idx += 1
        terminated = (self.current_idx >= len(self.unlabeled_x))
        info = {}
        
        if terminated:
            self.downstream_model.train_model((self.labeled_x, self.labeled_y), self.newly_pseudo_labeled_data)
            new_accuracy = self.downstream_model.evaluate_model(self.val_x, self.val_y)
            info['new_accuracy'] = new_accuracy
            
            accuracy_bonus = new_accuracy - self.baseline_accuracy
            reward += accuracy_bonus
            info['final_reward'] = self.episode_reward + accuracy_bonus
        
        next_state = self._get_state()
        return next_state, reward, terminated, False, info

    # -----------------------------
    # Reward function
    # -----------------------------
    def calculate_reward(self, pred_label, true_label, action, probs):
        """
        Reward function:
        - If action is skip (action == num_classes), return small negative reward
        - If action is a label, reward is +confidence if action matches true label, else -confidence
        """
        if action == self.num_classes:
            return -0.1  # Penalty for skipping
        confidence = probs[action].item()
        return confidence if action == true_label else -confidence
    
    def _get_state(self):
        if self.current_idx >= len(self.unlabeled_x):
            return np.zeros(self.observation_space.shape)
        
        img_data = self.unlabeled_x[self.current_idx]
        preds = self.downstream_model.get_predictions(img_data.unsqueeze(0))
        entropy = -torch.sum(preds * torch.log(preds + 1e-9))
        
        state = torch.cat([img_data, preds, entropy.unsqueeze(0)])
        return state.cpu().numpy()

    def render(self):
        pass

    def close(self):
        pass
    
    
# -----------------------------
# Downstream Model Class
# -----------------------------
class DownstreamModel(nn.Module):
    """A single class that combines the CNN architecture and its training/evaluation methods."""
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.to(device)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, labeled_data, pseudo_labeled_data, epochs=1):
        x_train, y_train = labeled_data
        
        if pseudo_labeled_data:
            pseudo_x = torch.stack([item[0] for item in pseudo_labeled_data])
            pseudo_y = torch.stack([item[1] for item in pseudo_labeled_data])
            x_train = torch.cat([x_train, pseudo_x])
            y_train = torch.cat([y_train, pseudo_y])

        if x_train.size(0) == 0:
            return  # Nothing to train on
        
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.train()
        for _ in range(epochs):
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                self.optimizer.zero_grad()
                out = self(x_b)
                loss = self.criterion(out, y_b.long())
                loss.backward()
                self.optimizer.step()

    def evaluate_model(self, x_data, y_data):
        self.eval()
        with torch.no_grad():
            x_data = x_data.to(self.device)
            y_data = y_data.to(self.device)
            preds = torch.argmax(self(x_data), dim=1)
            return accuracy_score(y_data.cpu(), preds.cpu())

    def get_predictions(self, x_data):
        self.eval()
        with torch.no_grad():
            x_data = x_data.to(self.device)
            logits = self(x_data)
            return F.softmax(logits, dim=1).squeeze()