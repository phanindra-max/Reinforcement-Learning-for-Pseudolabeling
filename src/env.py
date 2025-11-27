import gymnasium as gym
import torch
import numpy as np
from model import init_downstream_model
from utils import load_source_data
from config import *

# -----------------------------
# Main Environment Class
# -----------------------------
class PseudoLabelEnv(gym.Env):

    def __init__(self, num_labeled, num_unlabeled, num_validation):
        super().__init__()
        self.device = device
        self.use_true_labels = use_true_labels

        labeled_x, labeled_y, unlabeled_x, unlabeled_y, val_x, val_y, test_x, test_y, dataset_info = load_source_data(data_source, num_labeled, num_unlabeled, num_validation)

        self.labeled_x, self.labeled_y = labeled_x.to(self.device), labeled_y.to(self.device)
        self.unlabeled_x, self.unlabeled_y = unlabeled_x.to(self.device), unlabeled_y.to(self.device)
        self.val_x, self.val_y = val_x.to(self.device), val_y.to(self.device)
        self.test_x, self.test_y = test_x.to(self.device), test_y.to(self.device)

        self.num_classes = dataset_info['num_classes']
        self.feature_dim = dataset_info['feature_dim']
        self.input_shape = dataset_info['input_shape']

        state_dim = self.feature_dim + self.num_classes + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        self.action_space = gym.spaces.Discrete(self.num_classes + 1)

        # Initialize and train downstream model once
        self.downstream_model, self.initial_model_state, self.baseline_accuracy = init_downstream_model(model_name, self.input_shape, self.num_classes, self.feature_dim, self.device, (self.labeled_x, self.labeled_y), (self.val_x, self.val_y))
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
        self.unlabeled_y = self.unlabeled_y[perm]

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

        if self.use_true_labels:
            chosen_label = self.unlabeled_y[img_idx]
            self.newly_pseudo_labeled_data.append((self.unlabeled_x[img_idx], chosen_label))
        else:
            if action < self.num_classes:
                chosen_label = torch.tensor(action, device=self.device)
                self.newly_pseudo_labeled_data.append((self.unlabeled_x[img_idx], chosen_label))
            else: # skipped
                reward += self.calculate_reward(0.0, 0.0, skip_penalty=True)

        if len(self.newly_pseudo_labeled_data) and ((self.current_idx+1 >= len(self.unlabeled_x)) or (len(self.newly_pseudo_labeled_data) == 512)):
          
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