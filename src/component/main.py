from env import PseudoLabelEnv
from model import SimpleCNN, PolicyGradient
from tqdm import tqdm
import numpy as np
import torch
from component.utils import retrain_downstream_model, evaluate_model, ReplayBuffer, get_ssl_mnist, evaluate_model_accuracy


def main(EPISODES, NUM_LABELED, NUM_UNLABELED, NUM_VALIDATION):

    # 1. Create the Custom Environment
    env = PseudoLabelEnv(NUM_LABELED=NUM_LABELED, NUM_UNLABELED=NUM_UNLABELED, NUM_VALIDATION=NUM_VALIDATION)

    # 2. Get State and Action Dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3. RL Agent
    agent = PolicyGradient(state_dim=state_dim, num_actions=action_dim)

    # 4. Downstream model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # 5. Loading labeled and unlabeled data
    labeled_loader, unlabeled_loader = get_ssl_mnist(labeled_size=100, unlabeled_size=2000)


    print("Starting training with PyTorch agent...")
    for episode in tqdm(range(EPISODES), desc="Training Progress"):
        state, info = env.reset()
        terminated = False
        
        # Lists to store the trajectory for this episode
        sampled_states, sampled_actions, sampled_rewards = [], [], []

        while not terminated:
            action = agent.policy(state)
            next_state, reward, terminated, _, info = env.step(action)
            
            # Store the experience
            sampled_states.append(state)
            sampled_actions.append(action)
            sampled_rewards.append(reward)
            
            state = next_state

        # After the episode is done, update the agent
        # We use np.vstack to stack the states into a single NumPy array
        agent.update(np.vstack(sampled_states), sampled_actions, sampled_rewards)

        # Log results from the info dict
        if 'final_reward' in info:
            print(f"\nEpisode {episode+1}: Final Reward = {info['final_reward']:.4f}, New Accuracy = {info['new_accuracy']:.4f}")

    env.close()
    print("Training finished.")

if __name__ == "__main__":


    # --- Hyperparameters ---
    EPISODES = 50
    # Data parameters for your environment
    NUM_LABELED = 1000
    NUM_UNLABELED = 4000 # The agent will process this many samples per episode
    NUM_VALIDATION = 1000

    main(EPISODES, NUM_LABELED=NUM_LABELED, NUM_UNLABELED=NUM_UNLABELED, NUM_VALIDATION=NUM_VALIDATION)