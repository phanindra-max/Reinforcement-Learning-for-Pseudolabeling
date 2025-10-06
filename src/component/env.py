import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch.nn.functional import softmax
from src.component.utils import compute_reward

class PseudoLabelEnv(gym.Env):
    """
    Gym-style environment for RL-based pseudo-labeling.
    Each step corresponds to assigning a pseudo-label to one unlabeled sample.
    State = model prediction probabilities + entropy (uncertainty)
    Reward = based on match between action and true label, weighted by confidence
    """

    def __init__(self, unlabeled_loader, model, device, num_classes=10, threshold=0):
        super().__init__()

        self.unlabeled_loader = unlabeled_loader
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.threshold = threshold

        # Gym spaces
        self.action_space = spaces.Discrete(num_classes)  # select a class label
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_classes + 1,),  # probs + entropy
            dtype=np.float32
        )

        # Initialize episode variables
        self.data_iter = None
        self.current_sample = None
        self.done = False

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment for a new episode.
        Returns initial state and info.
        """
        super().reset(seed=seed)
        self.data_iter = iter(self.unlabeled_loader)
        self.done = False
        self.current_sample = None
        state = self._get_next_state()
        info = {}
        return state, info

    def _get_next_state(self):
        """
        Loads the next sample and returns state vector (probs + entropy)
        """
        try:
            x, y = next(self.data_iter)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = softmax(logits, dim=1)  # shape [B, num_classes]
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)  # [B,1]

            # Flatten probs + entropy into a single vector
            state_vector = torch.cat([probs.view(-1), entropy.view(-1)]).cpu().numpy()

            self.current_sample = (x, y, probs.cpu().numpy())
            return state_vector

        except StopIteration:
            self.done = True
            self.current_sample = None
            return None

    def step(self, action):
        """
        Perform a pseudo-labeling action.
        Returns: next_state, reward, terminated, truncated, info
        """
        # If no current sample, episode is done
        if self.current_sample is None:
            return None, 0, True, False, {}

        x, true_label, probs = self.current_sample
        probs = probs[0]  # shape: [num_classes]

        # Apply confidence threshold
        if action != -1 and probs[action] < self.threshold:
            action = -1

        # Compute reward
        reward = 0
        if action != -1:
            pred_label = np.argmax(probs)
            reward = compute_reward(pred_label, true_label.item(), action, probs)

        # Move to next sample
        next_state = self._get_next_state()

        terminated = self.done
        truncated = False

        # Info dict for debugging
        info = {
            "true_label": true_label.item() if true_label is not None else None,
            "pred_label": np.argmax(probs) if probs is not None else None,
            "confidence": probs[action] if action != -1 else 0,
            "action_taken": action
        }

        return next_state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


