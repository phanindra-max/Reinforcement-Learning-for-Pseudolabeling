import gymnasium as gym
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score

class PseudoLabelEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.current_index = 0
        state = np.zeros(self.observation_space.shape, dtype = np.float32)
        info = {}

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        reward = 0
        if action == 3: # Placeholder: let's say action '3' is a good label
            reward = 1
        else:
            reward = -0.01 # Small penalty for other labels

   
        terminated = True  # Episode ends after one step (one labeling action)
        truncated = False  # we can add, if required

        # in the reset() function. Here, we can return the same state or a dummy one.
        observation = self._state

        return observation, reward, terminated, truncated

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

    def render(self):
        pass

    def close(self):
        pass
    
    