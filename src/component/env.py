import gymnasium as gym
import numpy as np

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

    def render(self):
        pass

    def close(self):
        pass