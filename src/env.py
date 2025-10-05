import torch
import numpy as np
from torch.nn.functional import softmax
from src.component.utils import compute_reward

class PseudoLabelEnv:
    def __init__(self, unlabeled_loader, model, device, threshold=0):
        '''
        Initilizes:
        - Unlabeled data iterator
        - Downstream model
        - Confidence threshold
        - DQN-friendly state space (probs + entropy)
        '''
        self.unlabeled_loader = unlabeled_loader
        self.model = model
        self.device = device
        self.threshold = threshold
        self.reset()

    def reset(self):
        '''
        - Prepares environment for a new episode
        - Returns the first state (features of sample)
        '''
        self.data_iter = iter(self.unlabeled_loader)
        self.done = False
        self.current_sample = None
        return self._get_next_state()

    def _get_next_state(self):
        '''
        - Loads the next sample
        - Computes:
            - probs: softmax output
            - entropy: uncertainty of prediction
        - Combines into a feature vector = RL state
        '''
        try:
            x, y = next(self.data_iter)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = softmax(logits, dim=1)  # shape [B, num_classes]
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)  # shape [B,1]

            state_vector = torch.cat([
                probs.view(-1),
                entropy.view(-1)
            ]).cpu().numpy()

            self.current_sample = (x, y, probs.cpu().numpy())
            return state_vector, y
        except StopIteration:
            self.done = True
            self.current_sample = None
            return None, None

    def step(self, action):
        '''
        - If confidence is high, accepts action
        - Computes:
            - pred_label using CNN
            - Reward = based on match between action and true label (and confidence)
        - Moves to next sample
        - Returns: next_state, reward, terminated, truncated
        '''
        if self.current_sample is None:
            return None, 0, True, False

        x, true_label, probs = self.current_sample
        probs = probs[0]

        # confidence threshold
        #if action != -1 and probs[action] < self.threshold:
        #    action = -1  # skip if confidence low

        reward = 0
        if action != -1:
            pred_label = np.argmax(probs)
            reward = compute_reward(pred_label, true_label.item(), action, probs)

        next_state, _ = self._get_next_state()
        terminated = self.done
        truncated = False
        #print(f"[STEP] Action={action}, True={true_label.item()}, Reward={reward:.3f}, Confidence={probs[action] if action != -1 else 'SKIPPED'}")
        return next_state, reward, terminated, truncated

    def close(self):
        pass
