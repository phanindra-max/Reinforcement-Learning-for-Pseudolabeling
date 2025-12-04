from env import PseudoLabelEnv
from model import PolicyGradient
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
from config import *
import os

def main():
    writer = SummaryWriter(f"runs/{run_name}")
    env = PseudoLabelEnv(num_labeled, num_unlabeled, num_validation)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    agent = PolicyGradient(state_dim=state_dim,num_actions=action_dim)
    best_error_rate = float('inf')

    for episode in tqdm(range(episodes), desc="Training Progress"):
        state, info = env.reset()
        sampled_states, sampled_actions, sampled_rewards, terminated = [], [], [], False
        while not terminated:
            action = agent.policy(state)
            next_state, reward, terminated, _, info = env.step(action)
            sampled_states.append(state)
            sampled_actions.append(action)
            sampled_rewards.append(reward)
            state = next_state

        policy_loss, value_loss = agent.update(np.vstack(sampled_states), sampled_actions, sampled_rewards)
        episode_reward = sum(sampled_rewards)  # Sum of step rewards
        if 'final_reward' in info:
            final_reward, error_rate = info['final_reward'], info['error_rate']
            if error_rate < best_error_rate:
                best_error_rate = error_rate
                env.downstream_model.save_model(name='best_downstream', save_dir=save_dir)
                agent.save_model(name='best_policy_gradient', save_dir=save_dir)
                
    env.close()
    writer.close()

if __name__ == "__main__":
    main()