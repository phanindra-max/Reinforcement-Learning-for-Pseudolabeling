import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os



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
            save_dir: Directory to save models (default: src/trained_models)
        '''
        if save_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_path, 'trained_models')
        
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.pi.state_dict(), os.path.join(save_dir, f'{name}_policy.pth'))
        torch.save(self.v.state_dict(), os.path.join(save_dir, f'{name}_value.pth'))
        print(f"Policy gradient models saved to {save_dir}/{name}_policy.pth and {save_dir}/{name}_value.pth")
    
    def load_model(self, name='policy_gradient', save_dir=None):
        '''
        Load policy and value networks
        
        Args:
            name: Base name for saved models
            save_dir: Directory to load models from (default: src/trained_models)
        '''
        if save_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_path, 'trained_models')
        
        self.pi.load_state_dict(torch.load(os.path.join(save_dir, f'{name}_policy.pth')))
        self.v.load_state_dict(torch.load(os.path.join(save_dir, f'{name}_value.pth')))
        print(f"Models loaded from {save_dir}")

