from env import PseudoLabelEnv
from model import SimpleCNN, PolicyGradient
from tqdm import tqdm
import numpy as np

def main(EPISODES, NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION):
    env = PseudoLabelEnv(NUM_LABELED=NUM_LABELED, NUM_UNLABELED=NUM_UNLABELED, NUM_VALIDATION=NUM_VALIDATION)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PolicyGradient(state_dim=state_dim, num_actions=action_dim)
    for episode in tqdm(range(EPISODES), desc="Training Progress"):
        state, info = env.reset()
        terminated = False
        sampled_states, sampled_actions, sampled_rewards = [], [], []
        while not terminated:
            action = agent.policy(state)
            next_state, reward, terminated, _, info = env.step(action)
            
            # Store the experience
            sampled_states.append(state)
            sampled_actions.append(action)
            sampled_rewards.append(reward)
            
            state = next_state

        agent.update(np.vstack(sampled_states), sampled_actions, sampled_rewards)

    env.close()

if __name__ == "__main__":
    # --- Hyperparameters ---
    EPISODES = 50
    # Data parameters for your environment
    NUM_LABELED = 1000
    NUM_UNLABELED = 4000 # The agent will process this many samples per episode
    NUM_VALIDATION = 1000

    main(EPISODES, NUM_LABELED=NUM_LABELED, NUM_UNLABELED=NUM_UNLABELED, NUM_VALIDATION=NUM_VALIDATION)