import gymnasium as gym
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from config import *
import os


# -----------------------------
# Main Environment Class
# -----------------------------
class PseudoLabelEnv(gym.Env):
    """
    A reinforcement learning environment for pseudo-labeling unlabeled data.
    
    This environment allows an RL agent to decide whether to assign pseudo-labels
    to unlabeled images or skip them. The agent is rewarded based on the confidence
    and correctness of its labeling decisions.
    
    Attributes:
        device (torch.device): Device for tensor computations (CPU or CUDA)
        labeled_x (torch.Tensor): Features of labeled training data
        labeled_y (torch.Tensor): Labels of labeled training data
        unlabeled_x (torch.Tensor): Features of unlabeled data
        unlabeled_y_true (torch.Tensor): True labels of unlabeled data (for evaluation)
        val_x (torch.Tensor): Features of validation data
        val_y (torch.Tensor): Labels of validation data
        num_classes (int): Number of classification classes
        feature_dim (int): Dimensionality of input features
        observation_space (gym.spaces.Box): State space definition
        action_space (gym.spaces.Discrete): Action space definition
        downstream_model (DownstreamModel): Classification model
        baseline_accuracy (float): Initial validation accuracy in an Episode
    """

    def __init__(self, NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION):
        """
        Initialize the pseudo-labeling environment.
        
        Args:
            NUM_LABELED (int): Number of labeled samples to use for initial training
            NUM_UNLABELED (int): Number of unlabeled samples for pseudo-labeling
            NUM_VALIDATION (int): Number of validation samples for evaluation
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        labeled_data, unlabeled_data, validation_data = self._load_data(NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION)

        self.labeled_x, self.labeled_y, self.unlabeled_x, self.unlabeled_y_true = \
            labeled_data[0].to(self.device), labeled_data[1].to(self.device), \
            unlabeled_data[0].to(self.device), unlabeled_data[1].to(self.device)
        self.val_x, self.val_y = validation_data[0].to(self.device), validation_data[1].to(self.device)

        self.num_classes = len(torch.unique(labeled_data[1]))
        self.feature_dim = self.labeled_x.shape[1]
        state_dim = self.feature_dim + self.num_classes + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        self.action_space = gym.spaces.Discrete(self.num_classes + 1)

        # Initialize and train downstream model once
        self.downstream_model = DownstreamModel(self.device)
        self.downstream_model.train_model((self.labeled_x, self.labeled_y), [])
        self.initial_model_state = self.downstream_model.state_dict()  # Save initial weights
        self.baseline_accuracy = self.downstream_model.evaluate_model(self.val_x, self.val_y)

    # -----------------------------
    # Load Data
    # -----------------------------
    def _load_data(self, NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION):
        """
        Load and split MNIST dataset into labeled, unlabeled, and validation sets.
        
        Args:
            NUM_LABELED (int): Number of samples for labeled training set
            NUM_UNLABELED (int): Number of samples for unlabeled set
            NUM_VALIDATION (int): Number of samples for validation set
        
        Returns:
            tuple: Three tuples containing (features, labels) for labeled, unlabeled,
                   and validation datasets respectively. Features are flattened to 784-dim vectors.
        """
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
        """
        Reset the environment to start a new episode.
        
        Resets the current index, shuffles unlabeled data, clears pseudo-labeled data,
        and restores the downstream model to its initial trained state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options (not used)
        
        Returns:
            tuple: Initial state observation and empty info dictionary
        """
        super().reset(seed=seed)
        self.current_idx = 0
        self.newly_pseudo_labeled_data = []
        self.episode_reward = 0.0

        perm = torch.randperm(self.unlabeled_x.size(0))
        self.unlabeled_x = self.unlabeled_x[perm]
        self.unlabeled_y_true = self.unlabeled_y_true[perm]

        # Restore downstream model to initial weights
        self.downstream_model.load_state_dict(self.initial_model_state)
        print("Initial (Episode-Start) Accuracy on Validation Data: ", self.baseline_accuracy)
        return self._get_state(), {}

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):
        """
        Execute one step in the environment based on the agent's action.
        
        Args:
            action (int): Action to take. If action < num_classes, assign that label
                         to current unlabeled sample. If action == num_classes, skip the sample.
        
        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
                - next_state (np.ndarray): Observation of next state
                - reward (float): Reward for the action taken
                - terminated (bool): Whether episode has ended
                - truncated (bool): Always False (not used)
                - info (dict): Additional information including final accuracy if terminated
        """
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
            info['error_rate'] = 1 - new_accuracy
            
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
        Calculate reward for the action taken on current unlabeled sample.
        
        Args:
            pred_label (int, optional): Model's predicted label for the sample
            true_label (int, optional): True label of the sample
            action (int): Action taken by agent (label assignment or skip)
            probs (torch.Tensor, optional): Probability distribution over classes
        
        Returns:
            float: Reward value. For skip action, returns -0.1. For labeling action,
                   returns +confidence if correct, -confidence if incorrect.
        """
        if action == self.num_classes:
            return -0.1  # Penalty for skipping
        confidence = probs[action].item()
        return confidence if action == true_label else -confidence
    

    # -----------------------------
    # Get Next State
    # -----------------------------
    def _get_state(self):
        """
        Construct the state observation for the current unlabeled sample.
        
        Returns:
            np.ndarray: State vector containing concatenation of:
                - Image features (784-dim)
                - Model's predicted probability distribution (num_classes-dim)
                - Prediction entropy (1-dim)
                Returns zero vector if all samples have been processed.
        """
        if self.current_idx >= len(self.unlabeled_x):
            return np.zeros(self.observation_space.shape)
        
        img_data = self.unlabeled_x[self.current_idx]
        preds = self.downstream_model.get_predictions(img_data.unsqueeze(0))
        entropy = -torch.sum(preds * torch.log(preds + 1e-9))
        
        state = torch.cat([img_data, preds, entropy.unsqueeze(0)])
        return state.cpu().numpy()

    def render(self):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Clean up environment resources (not implemented)."""
        pass
    
    
# -----------------------------
# Downstream Model Class
# -----------------------------
class DownstreamModel(nn.Module):
    """
    A CNN-based classification model for MNIST digit recognition.
    
    This model serves as the downstream task classifier that gets trained on
    labeled data and pseudo-labeled data. It combines the neural network
    architecture with training and evaluation methods.
    
    Architecture:
        - Conv2d(1->16, 5x5) + ReLU + MaxPool2d
        - Conv2d(16->32, 5x5) + ReLU + MaxPool2d
        - Fully Connected(1568->128) + ReLU
        - Fully Connected(128->10)
    
    Attributes:
        device (torch.device): Device for computations
        conv1, conv2 (nn.Conv2d): Convolutional layers
        pool (nn.MaxPool2d): Pooling layer
        fc1, fc2 (nn.Linear): Fully connected layers
        optimizer (optim.Adam): Optimizer for training
        criterion (nn.CrossEntropyLoss): Loss function
    """
    
    def __init__(self, device):
        """
        Initialize the downstream CNN model.
        
        Args:
            device (torch.device): Device to place model on (CPU or CUDA)
        """
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.optimizer = optim.Adam(self.parameters(), lr=DOWNSTREAM_LR)
        self.criterion = nn.CrossEntropyLoss()
        self.to(device)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784) or (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Logits for each class, shape (batch_size, 10)
        """
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, labeled_data, pseudo_labeled_data, epochs=DOWNSTREAM_EPOCHS):
        """
        Train the model on labeled and pseudo-labeled data.
        
        Combines labeled training data with pseudo-labeled data and trains
        the model for the specified number of epochs using mini-batch gradient descent.
        
        Args:
            labeled_data (tuple): Tuple of (features, labels) for labeled samples
            pseudo_labeled_data (list): List of (feature, label) tuples for pseudo-labeled samples
            epochs (int, optional): Number of training epochs. Defaults to DOWNSTREAM_EPOCHS from config.
        """
        x_train, y_train = labeled_data
        
        if pseudo_labeled_data:
            pseudo_x = torch.stack([item[0] for item in pseudo_labeled_data])
            pseudo_y = torch.stack([item[1] for item in pseudo_labeled_data])
            x_train = torch.cat([x_train, pseudo_x])
            y_train = torch.cat([y_train, pseudo_y])

        if x_train.size(0) == 0:
            return  # Nothing to train on
        
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
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
        """
        Evaluate model accuracy on given data.
        
        Args:
            x_data (torch.Tensor): Input features to evaluate
            y_data (torch.Tensor): True labels for evaluation
        
        Returns:
            float: Classification accuracy (between 0 and 1)
        """
        self.eval()
        with torch.no_grad():
            x_data = x_data.to(self.device)
            y_data = y_data.to(self.device)
            preds = torch.argmax(self(x_data), dim=1)
            return accuracy_score(y_data.cpu(), preds.cpu())

    def get_predictions(self, x_data):
        """
        Get probability distribution over classes for input data.
        
        Args:
            x_data (torch.Tensor): Input features, shape (batch_size, 784)
        
        Returns:
            torch.Tensor: Softmax probabilities for each class, shape (batch_size, 10)
                         or (10,) if batch_size=1
        """
        self.eval()
        with torch.no_grad():
            x_data = x_data.to(self.device)
            logits = self(x_data)
            return F.softmax(logits, dim=1).squeeze()
        
    def save_model(self, name='downstream', save_dir=None):
        """
        Save the downstream model state to disk.
        
        Args:
            name (str, optional): Base name for saved model file. Defaults to 'downstream'.
            save_dir (str, optional): Directory to save model. Defaults to 'src/trained_models'
                                     relative to the current file's directory.
        """
        if save_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_path, 'trained_models')
        
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.state_dict(), os.path.join(save_dir, f'{name}.pth'))
        print(f"Downstream model saved to {save_dir}/{name}.pth")