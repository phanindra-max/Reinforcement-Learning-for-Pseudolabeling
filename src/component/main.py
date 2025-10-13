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
    env = PseudoLabelEnv(NUM_LABELED=NUM_LABELED, NUM_UNLABELED=NUM_UNLABELED, NUM_VALIDATION=NUM_VALIDATION)

    # Get State and Action Dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # RL Agent
    agent = PolicyGradient(
        state_dim=state_dim,
        num_actions=action_dim,
        gamma=GAMMA,
        alpha_pi=ALPHA_PI,
        alpha_v=ALPHA_V,
        hidden_dim=HIDDEN_DIM,
        policy=POLICY_TYPE,
        device=env.device
    )

    # Initialize best accuracy tracking
    best_accuracy = float('-inf')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
    os.makedirs(save_dir, exist_ok=True)

    print("Starting training with PyTorch agent...")
    for episode in tqdm(range(EPISODES), desc="Training Progress"):
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
            episode_reward += info['new_accuracy'] - env.baseline_accuracy  # Include accuracy bonus
            writer.add_scalar("Reward/Final", info['final_reward'], episode)
            writer.add_scalar("Accuracy/New", info['new_accuracy']*100, episode)
            writer.add_scalar("Error Rate", info['error_rate']*100, episode)
            print(f"Episode {episode+1}: Final Reward = {info['final_reward']:.4f}, "
                  f"New Accuracy = {info['new_accuracy']*100:.4f}, Error Rate = {info['error_rate']*100:.4f}, "
                  f"Policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}\n")
        
            # Check if this is the best accuracy
            if info['new_accuracy'] > best_accuracy:
                best_accuracy = info['new_accuracy']
                print(f"New best accuracy: {best_accuracy*100:.4f}% - Saving models...")
                # Save downstream model
                env.downstream_model.save_model(name='best_downstream', save_dir=save_dir)
                # Save policy gradient models
                agent.save_model(name='best_policy_gradient', save_dir=save_dir)

        # Log policy and value losses
        writer.add_scalar("Loss/Policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)
        writer.add_scalar("Reward/Episode", episode_reward, episode)

    env.close()
    writer.close()  # Close the TensorBoard writer
    print(f"Training finished. Best accuracy achieved: {best_accuracy*100:.4f}%")


if __name__ == "__main__":
    main()