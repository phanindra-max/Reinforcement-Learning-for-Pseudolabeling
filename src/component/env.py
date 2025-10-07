import gymnasium as gym
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.nn.functional import softmax


# -----------------------------
# Main Environment Class
# -----------------------------
class PseudoLabelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data: load_data now returns true labels for unlabeled data
        labeled_data, unlabeled_data, validation_data = self.load_data(NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION)

        self.labeled_x, self.labeled_y, self.unlabeled_x, self.unlabeled_y_true = \
            labeled_data[0].to(self.device), labeled_data[1].to(self.device), \
            unlabeled_data[0].to(self.device), unlabeled_data[1].to(self.device)
        self.val_x, self.val_y = validation_data[0].to(self.device), validation_data[1].to(self.device)

        # Environment Specs
        self.num_classes = 10
        self.feature_dim = self.labeled_x.shape[1]

        # State: [image_features, softmax_predictions, entropy]
        state_dim = self.feature_dim + self.num_classes + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))

        # Action: [0..9] for labels, 10 for 'skip'
        self.action_space = gym.spaces.Discrete(self.num_classes + 1)

    # -----------------------------
    # Load Data
    # -----------------------------
    def load_data(self, NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION):
        transform = transforms.ToTensor()
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        # The data is already in a 28x28 format from the ToTensor() transform.
        all_data = mnist_train.data.float() / 255.0
        all_targets = mnist_train.targets

        # First split off validation
        data_temp, val_data, targets_temp, val_targets = train_test_split(
            all_data, all_targets, test_size=NUM_VALIDATION, stratify=all_targets, random_state=42
        )

        # Now split labeled vs unlabeled
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
        self.data_iter = iter(self.unlabeled_loader)
        self.done = False
        self.current_sample = None

        state, _ = self._get_next_state()
        info = {}
        return state, info

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):
        # If no current sample, episode done
        if self.current_sample is None:
            return None, 0, True, False, {}

        x, true_label, probs = self.current_sample
        probs = probs[0]  # shape: [num_classes]

        # Confidence threshold: skip low-confidence samples
        if action != -1 and probs[action] < self.threshold:
            action = -1

        # Reward computation
        reward = 0
        if action != -1:
            pred_label = np.argmax(probs)
            reward = compute_reward(pred_label, true_label.item(), action, probs)

        # Next state
        next_state, _ = self._get_next_state()

        terminated = self.done
        truncated = False

        info = {
            "true_label": true_label.item() if true_label is not None else None,
            "pred_label": np.argmax(probs) if probs is not None else None,
            "confidence": probs[action] if action != -1 else 0,
            "action_taken": action,
        }

        return next_state, reward, terminated, truncated, info

    # -----------------------------
    # Temporary reward function
    # -----------------------------
    def calculate_reward(self, pred_label, true_label, action, probs):
        """
        Basic reward:
        +1 if prediction and action match true label
        -1 otherwise
        Weighted by confidence
        """
        if action == -1:
            return -0.1  # penalty for skipping
        confidence = probs[action]
        return confidence if pred_label == true_label else -confidence
    
    
    def _get_state(self):
        if self.current_idx >= len(self.unlabeled_x):
            return np.zeros(self.observation_space.shape)

        with torch.no_grad():
            img_data = self.unlabeled_x[self.current_idx]
            logits = self.downstream_model(img_data.unsqueeze(0))
            preds = torch.softmax(logits, dim=1).squeeze()
            entropy = -torch.sum(preds * torch.log(preds + 1e-9))

            # Concatenate features, predictions, and entropy to form the state
            state = torch.cat([img_data, preds, entropy.unsqueeze(0)])
            return state.cpu().numpy()

    def render(self):
        pass

    def close(self):
        pass
    
    