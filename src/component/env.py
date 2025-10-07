import gymnasium as gym
import torch
import numpy as np
from torch.nn.functional import softmax

# -----------------------------
# Temporary reward function
# -----------------------------
def compute_reward(pred_label, true_label, action, probs):
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


# -----------------------------
# Main Environment Class
# -----------------------------
class PseudoLabelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, unlabeled_loader, model, device, threshold=0.5):
        super().__init__()
        self.unlabeled_loader = unlabeled_loader
        self.model = model
        self.device = device
        self.threshold = threshold

        # RL-compatible spaces
        num_classes = 10  # <-- adjust this to your dataset
        # state: class probs + entropy (1D vector)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(num_classes + 1,), dtype=np.float32
        )
        # action: choose class index or -1 to skip
        self.action_space = gym.spaces.Discrete(num_classes)

        self.reset()

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

    def _get_next_state(self):
        try:
            x, y = next(self.data_iter)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)

            # Combine probabilities + entropy into a single state vector
            state_vector = torch.cat([probs.view(-1), entropy.view(-1)]).cpu().numpy()

            self.current_sample = (x, y, probs.cpu().numpy())
            return state_vector.astype(np.float32), y

        except StopIteration:
            self.done = True
            self.current_sample = None
            return np.zeros(self.observation_space.shape, dtype=np.float32), None

    def render(self):
        pass

    def close(self):
        pass
    
    