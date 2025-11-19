import gymnasium as gym
import torch
import numpy as np
from utils import load_source_data
from model import DownstreamModel

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

    def __init__(self, num_labeled, num_unlabeled, num_validation):
        """
        Initialize the pseudo-labeling environment.
        Args:
            num_labeled (int): Number of labeled samples to use for initial training
            num_unlabeled (int): Number of unlabeled samples for pseudo-labeling
            num_validation (int): Number of validation samples for evaluation
        """

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        labeled_data, unlabeled_data, validation_data = load_source_data(num_labeled, num_unlabeled, num_validation)

        self.labeled_x, self.labeled_y, self.unlabeled_x, self.unlabeled_y_true = \
            labeled_data[0].to(self.device), labeled_data[1].to(self.device), \
            unlabeled_data[0].to(self.device), unlabeled_data[1].to(self.device)
        self.val_x, self.val_y = validation_data[0].to(self.device), validation_data[1].to(self.device)

        self.num_classes = 10
        self.feature_dim = self.labeled_x.shape[1]
        state_dim = self.feature_dim + self.num_classes + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        self.action_space = gym.spaces.Discrete(self.num_classes + 1)

        # Initialize and train downstream model once
        self.downstream_model = DownstreamModel(self.device)
        self.downstream_model.train_model((self.labeled_x, self.labeled_y), [])
        self.initial_model_state = self.downstream_model.state_dict()  # Save initial weights
        self.baseline_accuracy = self.downstream_model.evaluate_model(self.val_x, self.val_y)
        self.baseline_error_rate = 1.0 - self.baseline_accuracy
    
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
        self.all_pseudo_labeled_data = []
        self.episode_reward = 0.0

        perm = torch.randperm(self.unlabeled_x.size(0))
        self.unlabeled_x = self.unlabeled_x[perm]
        self.unlabeled_y_true = self.unlabeled_y_true[perm]

        # Restore the downstream model to initial weights
        self.downstream_model.load_state_dict(self.initial_model_state)
        print("Initial (Episode-Start) Error Rate on Validation Data: {:.2f}%".format(self.baseline_error_rate * 100))
        return self.get_next_state(), {}

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):
        """
        Execute one step in the environment based on the agent's action.
        """
        img_idx = self.current_idx
        reward = 0.0
      
        # Handle action (labeling or skipping)
        if action < self.num_classes:
            self.newly_pseudo_labeled_data.append(
                (self.unlabeled_x[img_idx], torch.tensor(action, device=self.device))
            )            
        else:  # Action is skip (action == num_classes)
            reward = self.calculate_reward(0.0, 0.0, skip_penalty=True)

        if len(self.newly_pseudo_labeled_data) and ((self.current_idx+1 >= len(self.unlabeled_x)) or (len(self.newly_pseudo_labeled_data) == 32)):
          
            # Store the baseline error rate for comparison
            old_error_rate = self.baseline_error_rate
          
            # Reset the model to the initial state (trained only on labeled data)
            self.downstream_model.load_state_dict(self.initial_model_state)
          
            # Train the model: Labeled data + CURRENT batch of pseudo-labels
            self.downstream_model.train_model(
                (self.labeled_x, self.labeled_y),
                self.newly_pseudo_labeled_data
            )
          
            # Evaluate new error rate
            new_accuracy = self.downstream_model.evaluate_model(self.val_x, self.val_y)
            new_error_rate = 1.0 - new_accuracy
          
            # Use separate method to calculate reward
            reward += self.calculate_reward(old_error_rate, new_error_rate)
          
            # Move current batch to all_pseudo_labeled_data
            self.all_pseudo_labeled_data.extend(self.newly_pseudo_labeled_data)
            # Clear the buffer for the next batch
            self.newly_pseudo_labeled_data = []

        self.episode_reward += reward
        self.current_idx += 1
        terminated = (self.current_idx >= len(self.unlabeled_x))
        info = {}
      
        if terminated:
            # 1. Reset model to the baseline state
            self.downstream_model.load_state_dict(self.initial_model_state)
          
            # 2. Train the model on the ENTIRE set of pseudo-labels collected in the episode
            if self.all_pseudo_labeled_data:
                self.downstream_model.train_model(
                    (self.labeled_x, self.labeled_y),
                    self.all_pseudo_labeled_data
                )
            # 3. Evaluate and report final metrics
            new_accuracy = self.downstream_model.evaluate_model(self.val_x, self.val_y)
            new_error_rate = 1.0 - new_accuracy
          
            info['new_accuracy'] = new_accuracy
            info['error_rate'] = new_error_rate
            info['final_reward'] = self.episode_reward
          
            print(f"Episode Terminated. Final Error Rate: {new_error_rate*100:.4f}%.")
        
        next_state = self.get_next_state()
        return next_state, reward, terminated, False, info
    
    # -----------------------------
    # Reward function
    # -----------------------------
    def calculate_reward(self, old_error_rate, new_error_rate, skip_penalty=False):
        """
        Calculate reward based on error rate reduction.
        
        Args:
            old_error_rate (float): Error rate before training with new pseudo-labels
            new_error_rate (float): Error rate after training with new pseudo-labels
            skip_penalty (bool): Whether to apply a small penalty for skipping (default: False)
        
        Returns:
            float: Reward value
        """
        reward = old_error_rate - new_error_rate
        if skip_penalty:
            reward -= 0.01  # Small penalty for skipping
        return reward
    

    # -----------------------------
    # Get Next State
    # -----------------------------
    def get_next_state(self):
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