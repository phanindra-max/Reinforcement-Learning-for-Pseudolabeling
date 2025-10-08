from env import PseudoLabelEnv
from model import PolicyGradient
from tqdm import tqdm
import numpy as np
import torch
from utils import retrain_downstream_model, get_ssl_mnist


def main(episodes, num_labeled, num_unlabeled, num_validation):

    # 1. Create the Custom Environment
    env = PseudoLabelEnv(num_labeled=num_labeled, num_unlabeled=num_unlabeled, num_validation=num_validation)

    # 2. Get State and Action Dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3. RL Agent
    agent = PolicyGradient(state_dim=state_dim, num_actions=action_dim)

    # 4. Downstream model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 5. Loading labeled and unlabeled data
    labeled_loader, unlabeled_loader = get_ssl_mnist(labeled_size=100, unlabeled_size=2000)

    pseudo_x, pseudo_y = [], []
    SKIP_ACTION = env.action_space.n - 1

    print("Starting training with PyTorch agent...")
    for episode in tqdm(range(episodes), desc="Training Progress"):
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

            if action != SKIP_ACTION:
                pseudo_x.append(torch.tensor(state[:784].copy()).float().view(784))
                pseudo_y.append(torch.tensor(action).long())

        # After the episode is done, update the agent
        # We use np.vstack to stack the states into a single NumPy array
        agent.update(np.vstack(sampled_states), sampled_actions, sampled_rewards)

        if len(pseudo_x) > 0:
            pseudo_data = list(zip(pseudo_x, pseudo_y))
            retrain_downstream_model(env.downstream_model, pseudo_data, device)

        # Log results from the info dict
        if 'final_reward' in info:
            print(f"\nEpisode {episode+1}: Final Reward = {info['final_reward']:.4f}, New Accuracy = {info['new_accuracy']:.4f}")

        print(f"Episode {episode + 1} done. Total Reward: {sum(sampled_rewards):.2f}, Pseudo-labels: {len(pseudo_y)}")
    env.close()
    print("Training finished.")

if __name__ == "__main__":

    episodes = 3
    num_labeled = 100
    num_unlabeled = 400
    num_validation = 100

    '''
    # --- Hyperparameters ---
    episodes = 50
    # Data parameters for your environment
    num_labeled = 1000
    num_unlabeled = 4000 # The agent will process this many samples per episode
    num_validation = 1000
    '''

    main(episodes, num_labeled=num_labeled, num_unlabeled=num_unlabeled, num_validation=num_validation)