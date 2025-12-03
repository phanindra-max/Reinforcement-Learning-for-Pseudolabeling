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
    """
    A reinforcement learning environment for pseudo-labeling unlabeled data.

    This environment allows an RL agent to decide whether to assign pseudo-labels to unlabeled images or skip them. The agent is rewarded based on the confidence and correctness of its labeling decisions.

    Attributes:
        device (torch.device): Device used for tensor computation.
        labeled_x (torch.Tensor): Labeled training images.
        labeled_y (torch.Tensor): True labels for labeled data.
        unlabeled_x (torch.Tensor): Unlabeled samples the RL agent acts on.
        val_x (torch.Tensor): Validation images for evaluating error rate.
        val_y (torch.Tensor): Validation labels.
        num_classes (int): Number of output classes.
        downstream_model: Classifier used for pseudo-label training.
        baseline_error_rate (float): Initial validation error rate.
        observation_space (gym.spaces.Box): State vector space.
        action_space (gym.spaces.Discrete): Action space (classes + skip).
    """

    # -----------------------------
    # INIT
    # -----------------------------
    def __init__(self, num_labeled, num_unlabeled, num_validation):
        """
        Initialize the environment by loading the dataset, preparing labeled/unlabeled splits, and training the downstream classifier once on labeled data.

        Args:
           num_labeled (int): Number of labeled samples to load.
           num_unlabeled (int): Number of unlabeled samples used for RL.
           num_validation (int): Number of validation samples used to compute the error rate.
       """
        super().__init__()
        self.device = device
        self.use_true_labels = use_true_labels

        # Load labeled, unlabeled, validation sets and dataset metadata
        labeled_x, labeled_y, unlabeled_x, unlabeled_y, val_x, val_y, test_x, test_y, dataset_info = load_source_data(data_source, num_labeled, num_unlabeled, num_validation)

        # Move tensors to device
        self.labeled_x, self.labeled_y = labeled_x.to(self.device), labeled_y.to(self.device)
        self.unlabeled_x, self.unlabeled_y = unlabeled_x.to(self.device), unlabeled_y.to(self.device)
        self.val_x, self.val_y = val_x.to(self.device), val_y.to(self.device)

        self.num_classes = dataset_info['num_classes']
        self.feature_dim = dataset_info['feature_dim']
        self.input_shape = dataset_info['input_shape']

        # Initialize and pre-train the downstream model once on labeled data
        self.downstream_model, self.initial_model_state, self.baseline_accuracy = init_downstream_model(model_name, self.input_shape, self.num_classes, self.feature_dim, self.device, (self.labeled_x, self.labeled_y), (self.val_x, self.val_y))
        self.baseline_error_rate = 1.0 - self.baseline_accuracy

        #Define state and action spaces
        example_feats = self.downstream_model.extract_features(self.labeled_x[:1]).shape[1] # extracted feature vector dimension
        state_dim = example_feats + self.num_classes + 1 # size of [extracted features, class probabilities, entropy]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        self.action_space = gym.spaces.Discrete(self.num_classes + 1)

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        Resets:
            - current index and error rate
            - pseudo-label buffers
            - unlabeled data order (shuffle)
            - classifier to initial trained state

        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options (not used)

        Returns:
            tuple: Initial state observation and empty info dictionary
        """
        super().reset(seed=seed)
        self.current_idx = 0
        self.newly_pseudo_labeled_data = [] # current pseudo-label batch
        self.all_pseudo_labeled_data = [] # accumulated pseudo-label batch
        self.episode_reward = 0.0
        self.current_error_rate = self.baseline_error_rate

        # shuffle unlabeled data
        perm = torch.randperm(self.unlabeled_x.size(0))
        self.unlabeled_x = self.unlabeled_x[perm]
        self.unlabeled_y = self.unlabeled_y[perm]

        # Reset the downstream model to initial baseline state
        self.downstream_model.load_state_dict(self.initial_model_state)
        print("Initial (Episode-Start) Error Rate on Validation Data: {:.2f}%".format(self.baseline_error_rate * 100))
        return self.get_next_state(), {}

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):
        """
        - Execute one step in the environment based on the agent's action by processing one unlabeled sample.
        - The agent chooses whether to assign a pseudo-label (0–(num_classes-1)) or skip the sample (last action index) (immediately receive penalty reward).
        - Once enough pseudo-labels are collected or the dataset ends, the classifier is retrained and reward is computed.

        Args:
            action (int): Agent-selected action indicating class or skip.

        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
        """

        img_idx = self.current_idx
        reward = 0.0

        # if running baseline experiment with true labels
        if self.use_true_labels:
            chosen_label = self.unlabeled_y[img_idx]
            self.newly_pseudo_labeled_data.append((self.unlabeled_x[img_idx], chosen_label))
        else: # if running a pseudo labeling experiment
            if action < self.num_classes:
                chosen_label = torch.tensor(action, device=self.device)
                self.newly_pseudo_labeled_data.append((self.unlabeled_x[img_idx], chosen_label))
            else: # skip
                reward += self.calculate_reward(0.0, 0.0, skip_penalty=True)

        # if end of unlabeled data or batch full, train downstream model and calculate reward
        if len(self.newly_pseudo_labeled_data) and ((self.current_idx+1 >= len(self.unlabeled_x)) or (len(self.newly_pseudo_labeled_data) == 512)):
            old_error_rate = self.current_error_rate

            # combine previous + current pseudo-label batch
            combined_pseudo_labeled_data = self.all_pseudo_labeled_data + self.newly_pseudo_labeled_data
            # Train the model: Labeled data + combined batch of pseudo-labels
            self.downstream_model.train_model(
                (self.labeled_x, self.labeled_y),
                combined_pseudo_labeled_data
            )
          
            # Evaluate new error rate and calculate reward
            new_accuracy = self.downstream_model.evaluate_model(self.val_x, self.val_y)
            new_error_rate = 1.0 - new_accuracy
            reward += self.calculate_reward(old_error_rate, new_error_rate)

            self.current_error_rate = new_error_rate
          
            # Move current batch to all_pseudo_labeled_data
            self.all_pseudo_labeled_data.extend(self.newly_pseudo_labeled_data)
            self.newly_pseudo_labeled_data = [] # Clear the buffer for the next batch

        # update counters
        self.episode_reward += reward
        self.current_idx += 1
        terminated = (self.current_idx >= len(self.unlabeled_x))
        info = {}
      
        if terminated:
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
            old_error_rate (float): Error rate before training on current pseudo-label batch.
            new_error_rate (float): Error rate after retraining the classifier.
            skip_penalty (bool): Whether to apply a small negative penalty for skipping.

        Returns:
            float: The computed reward value.
        """

        reward = old_error_rate - new_error_rate
        if skip_penalty:
            reward -= 0.01  # Small penalty for skipping
        return reward
    

    # -----------------------------
    # Build / Get Next State
    # -----------------------------
    def get_next_state(self):
        """
        Construct the environment state for the agent.
        The state includes:
        - CNN feature embedding of current image
        - class probability vector
        - entropy of probability distribution

        Returns:
            np.ndarray: State vector used as RL observation.
        """

        if self.current_idx >= len(self.unlabeled_x):
            return np.zeros(self.observation_space.shape)

        with torch.no_grad():
            img_data = self.unlabeled_x[self.current_idx]
            # add batch dimension -> extract features -> remove batch dimension: extracted 1D feature vector
            feats = self.downstream_model.extract_features(img_data.unsqueeze(0)).squeeze(0)
            # add batch dimension -> get predictions: class probabilities
            preds = self.downstream_model.get_predictions(img_data.unsqueeze(0))
            entropy = -torch.sum(preds * torch.log(preds + 1e-9))
            state = torch.cat([feats, preds, entropy.unsqueeze(0)]) # build state vector

        return state.cpu().numpy()

    def render(self):
        """
        Render is not implemented as this environment has no visual output
        """
        pass

    def close(self):
        """
        Close the environment; no resources to clean up
        """
        pass