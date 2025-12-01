import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
from config import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18, ResNet18_Weights

# -----------------------------
# Downstream Model
# -----------------------------
class CNN_Classifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        input_channels = input_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=num_classes))
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), *self.input_shape)
        x = self.features(x)
        return self.classifier(x)

class MLP_Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=feature_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes))
    def forward(self, x):
        return self.net(x)

class ResNet_Classifier(nn.Module):
    def __init__(self, input_shape, num_classes, use_pretrained=True):
        super().__init__()
        input_channels = input_shape[0]
        weights = ResNet18_Weights.IMAGENET1K_V1 if (use_pretrained and input_channels == 3) else None
        backbone = resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=num_classes)
        self.backbone = backbone
        self.input_shape = input_shape

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), *self.input_shape)
        return self.backbone(x)

class DownstreamModel(nn.Module):
    def __init__(self, model_name, input_shape, num_classes, feature_dim, device):
        super().__init__()
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device

        if model_name == 'cnn':
            self.backbone = CNN_Classifier(input_shape, num_classes)
        elif model_name == 'mlp':
            self.backbone = MLP_Classifier(feature_dim, num_classes)
        elif model_name == 'resnet18':
            self.backbone = ResNet_Classifier(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.optimizer = optim.Adam(self.parameters(), lr=downstream_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.to(device)

    def forward(self, x):
        if self.model_name in ('cnn', 'resnet18') and x.dim() == 2:
            x = x.view(x.size(0), *self.input_shape)
        return self.backbone(x)

    def train_model(self, labeled_data, pseudo_labeled_data, epochs=downstream_epochs):
        """
        Train the model on labeled and pseudo-labeled data.

        Combines labeled training data with pseudo-labeled data and trains
        the model for the specified number of epochs using mini-batch gradient descent.

        Args:
            labeled_data (tuple): Tuple of (features, labels) for labeled samples
            pseudo_labeled_data (list): List of (feature, label) tuples for pseudo-labeled samples
            epochs (int, optional): Number of training epochs. Defaults to downstream_epochs from config.
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
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
            logits = self(x_data.to(self.device))
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
            labels = y_data.cpu().numpy().astype(int)
            return accuracy_score(labels, preds)

    # def get_predictions(self, x_data):
    #     """
    #     Get probability distribution over classes for input data.
    #
    #     Args:
    #         x_data (torch.Tensor): Input features, shape (batch_size, 784)
    #
    #     Returns:
    #         torch.Tensor: Softmax probabilities for each class, shape (batch_size, 10)
    #                      or (10,) if batch_size=1
    #     """
    #     self.eval()
    #     with torch.no_grad():
    #         x_data = x_data.to(self.device)
    #         logits = self(x_data)
    #         return F.softmax(logits, dim=1).squeeze()

    def extract_features(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), *self.input_shape)
        if self.model_name == 'cnn':
            feats = self.backbone.features(x)
            return feats.view(feats.size(0), -1)
        if self.model_name == 'resnet18':
            b = self.backbone.backbone
            x = b.conv1(x);
            x = b.bn1(x);
            x = b.relu(x);
            x = b.maxpool(x)
            x = b.layer1(x);
            x = b.layer2(x);
            x = b.layer3(x);
            x = b.layer4(x)
            x = b.avgpool(x)
            return torch.flatten(x, 1)
            # mlp fallback just flattens
        return x.view(x.size(0), -1)

    def predict_from_features(self, feats):
        if self.model_name == 'cnn':
            return self.backbone.classifier(feats)
        if self.model_name == 'resnet18':
            return self.backbone.backbone.fc(feats)
        return self.backbone.net(feats)

    def get_predictions(self, x_data):
        """Probability distribution using extracted features."""
        self.eval()
        with torch.no_grad():
            feats = self.extract_features(x_data.to(self.device))
            logits = self.predict_from_features(feats)
            return F.softmax(logits, dim=1).squeeze()

    def save_model(self, name='downstream', save_dir=None):
        """
        Save the downstream model state to disk.

        Args:
            name (str, optional): Base name for saved model file. Defaults to 'downstream'.
            save_dir (str, optional): Directory to save model. Defaults to 'src/trained_models_<model>_<dataset>'
                                     relative to the current file's directory.
        """
        if save_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_path, f"trained_models_{model_name}_{data_source}")

        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(save_dir, f'{name}.pth'))
        print(f"Downstream model saved to {save_dir}/{name}.pth")

    def load_model(self, name='downstream', save_dir=None):
        if save_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_path, f"trained_models_{model_name}_{data_source}")
        ckpt = os.path.join(save_dir, f'{name}.pth')
        state = torch.load(ckpt, map_location=self.device)
        self.load_state_dict(state)
        self.eval()
        print(f"Downstream model loaded from {ckpt}")

def init_downstream_model(model_name, input_shape, num_classes, feature_dim, device, labeled_data, val_data):
    print(f"\n{'=' * 70}")
    print("DOWNSTREAM MODEL INITIALIZATION")
    print(f"{'=' * 70}")
    print(f"Initializing downstream model: {model_name}\n")
    model = DownstreamModel(model_name, input_shape, num_classes, feature_dim, device)
    model.train_model(labeled_data, [])
    initial_state = model.state_dict()
    baseline_acc = model.evaluate_model(*val_data)
    return model, initial_state, baseline_acc

class PolicyNetwork(nn.Module):
    '''
    Policy network for PolicyGradient
    Maps state to action logits
    '''
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class ValueNetwork(nn.Module):
    '''
    Value network for PolicyGradient
    Maps state to value estimate
    '''
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class PolicyGradient:
    '''
    Policy Gradient with baseline (Actor-Critic)
    Converted from TensorFlow to PyTorch
    
    Args:
        state_dim: Dimension of the state space
        num_actions: Number of possible actions
        gamma: Discount factor for future rewards
        alpha_pi: Learning rate for policy network
        alpha_v: Learning rate for value network
        device: Device to run on ('cpu' or 'cuda')
        policy: Type of policy ('mlp' for multi-layer perceptron)
    '''
    
    def __init__(self, state_dim, num_actions, gamma=0.99, alpha_pi=1e-3, 
                 alpha_v=1e-3, device='cpu', policy='mlp', hidden_dim=64):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device
        
        if policy == 'mlp':
            self.pi = PolicyNetwork(state_dim, num_actions, hidden_dim=hidden_dim).to(self.device)
            self.v = ValueNetwork(state_dim, hidden_dim=hidden_dim).to(device)
            self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=alpha_pi)
            self.optimizer_v = torch.optim.Adam(self.v.parameters(), lr=alpha_v)
        else:
            raise ValueError(f"Policy type '{policy}' not supported. Use 'mlp'.")

    def policy(self, state):
        '''
        Select an action based on the current policy
        
        Args:
            state: Current state (numpy array or list)
        
        Returns:
            action: Selected action (integer)
        '''
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.pi(state_tensor)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            return action.item()

    def compute_advantage(self, sampled_states, sampled_rewards):
        '''
        Compute advantages using rewards-to-go and value estimates
        
        Args:
            sampled_states: List of states from episode
            sampled_rewards: List of rewards from episode
        
        Returns:
            rewards_to_go: Discounted future rewards
            values: Value estimates for each state
            adv: Advantage estimates (rewards_to_go - values)
        '''
        rewards_to_go, values, adv = [], [], []
        g = 0
        
        # Compute rewards-to-go and advantages in reverse order
        for state, reward in list(zip(sampled_states, sampled_rewards))[::-1]:
            g = self.gamma * g + reward
            rewards_to_go.append(g)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                value = self.v(state_tensor).item()
                values.append(value)
                a = g - value
                adv.append(a)
        
        # Reverse to get correct order
        return (
            torch.FloatTensor(list(rewards_to_go[::-1])).to(self.device),
            torch.FloatTensor(list(values[::-1])).to(self.device),
            torch.FloatTensor(list(adv[::-1])).to(self.device)
        )
    
    def update(self, sampled_states, sampled_actions, sampled_rewards):
        '''
        Update policy and value networks based on episode data
        
        Args:
            sampled_states: List of states from episode
            sampled_actions: List of actions taken
            sampled_rewards: List of rewards received
        '''
        # Compute advantages
        rewards_to_go, values, adv = self.compute_advantage(sampled_states, sampled_rewards)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(sampled_states)).to(self.device)
        actions_tensor = torch.LongTensor(sampled_actions).to(self.device)
        
        # Update policy network
        self.optimizer_pi.zero_grad()
        logits = self.pi(states_tensor)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        policy_loss = -torch.mean(action_log_probs * adv.detach())
        policy_loss.backward()
        self.optimizer_pi.step()
        
        # Update value network
        self.optimizer_v.zero_grad()
        value_preds = self.v(states_tensor).squeeze(1)
        value_loss = torch.mean((rewards_to_go - value_preds) ** 2)
        value_loss.backward()
        self.optimizer_v.step()
        
        return policy_loss.item(), value_loss.item()
    
    def save_model(self, name='policy_gradient', save_dir=None):
        '''
        Save policy and value networks
        
        Args:
            name: Base name for saved models
            save_dir: Directory to save models (default: src/trained_models_<model>_<dataset>)
        '''
        if save_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_path, f"trained_models_{model_name}_{data_source}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.pi.state_dict(), os.path.join(save_dir, f'{name}_policy.pth'))
        torch.save(self.v.state_dict(), os.path.join(save_dir, f'{name}_value.pth'))
        print(f"Policy gradient models saved to {save_dir}/{name}_policy.pth and {save_dir}/{name}_value.pth")
    
    def load_model(self, name='policy_gradient', save_dir=None):
        '''
        Load policy and value networks
        
        Args:
            name: Base name for saved models
            save_dir: Directory to load models from (default: src/trained_models_<model>_<dataset>)
        '''
        if save_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_path, f"trained_models_{model_name}_{data_source}")
        
        self.pi.load_state_dict(torch.load(os.path.join(save_dir, f'{name}_policy.pth'), map_location=self.device))
        self.pi.eval()
        self.v.load_state_dict(torch.load(os.path.join(save_dir, f'{name}_value.pth'), map_location=self.device))
        self.v.eval()
        print(f"Models loaded from {save_dir}")

