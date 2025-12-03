from env import PseudoLabelEnv
from model import PolicyGradient
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
from config import *
import os

def main():
    # Initialize TensorBoard writer
    run_name = f"pseudo_labeling_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Create the Custom Environment
    env = PseudoLabelEnv(num_labeled=num_labeled, num_unlabeled=num_unlabeled, num_validation=num_validation)

    # Get State and Action Dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # RL Agent
    agent = PolicyGradient(
        state_dim=state_dim,
        num_actions=action_dim,
        gamma=gamma,
        alpha_pi=alpha_pi,
        alpha_v=alpha_v,
        hidden_dim=hidden_dim,
        policy=policy_type,
        device=env.device
    )

    # Initialize the best error rate tracking
    best_error_rate = float('inf')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trained_models_{model_name}_{data_source}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"{'=' * 70}")
    print("TRAINING LOOP")
    print(f"{'=' * 70}")
    print("Starting training with PyTorch agent...")
    for episode in tqdm(range(episodes), desc="Training Progress"):
        state, info = env.reset()
        terminated = False
        
        # Log baseline accuracy
        writer.add_scalar("Accuracy/Baseline", env.baseline_accuracy, episode)
        
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
        policy_loss, value_loss = agent.update(np.vstack(sampled_states), sampled_actions, sampled_rewards)

        # Log results from the info dict and episode metrics
        episode_reward = sum(sampled_rewards)  # Sum of step rewards
        if 'final_reward' in info:
            final_reward = info['final_reward']
            new_accuracy = info['new_accuracy']
            error_rate = info['error_rate']

            # Log the sum of the trajectory rewards for comparison
            writer.add_scalar("Reward/Episode", episode_reward, episode)
            writer.add_scalar("Accuracy/New", new_accuracy*100, episode)
            writer.add_scalar("Error Rate", error_rate*100, episode)
            
            print(f"Episode {episode+1}: Final Reward = {final_reward:.4f}, "
                  f"New Accuracy = {new_accuracy*100:.4f}, Error Rate = {error_rate*100:.4f}, "
                  f"policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}\n")
        
            # Check if this is the best error rate
            if error_rate < best_error_rate:
                best_error_rate = error_rate
                print(f"New best error rate: {best_error_rate*100:.4f}% - Saving models...")
                env.downstream_model.save_model(name='best_downstream', save_dir=save_dir) # Save downstream model
                agent.save_model(name='best_policy_gradient', save_dir=save_dir) # Save policy gradient models

        # Log policy and value losses
        writer.add_scalar("Loss/policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)

    print(f"{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")

    env.close()
    writer.close()
    print(f"Training finished. Best error rate achieved: {best_error_rate*100:.4f}%")

if __name__ == "__main__":
    main()