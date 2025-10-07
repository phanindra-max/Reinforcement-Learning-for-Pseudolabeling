"""
Example of using PolicyGradient for pseudo-labeling in semi-supervised learning

This script demonstrates how to use the PolicyGradient class instead of DQN
for the pseudo-labeling task in your project.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from model import SimpleCNN, PolicyGradient
from env import PseudoLabelEnv
from component.utils import get_ssl_mnist, retrain_downstream_model, evaluate_model_accuracy

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
labeled_loader, unlabeled_loader = get_ssl_mnist(labeled_size=100, unlabeled_size=2000)

# Initialize CNN
model = SimpleCNN().to(device)
env = PseudoLabelEnv(unlabeled_loader, model, device)

# Initialize PolicyGradient
state_dim = 11  # 10 class probabilities + 1 entropy
num_actions = 11  # 10 labels + 1 skip action
gamma = 0.99
alpha_pi = 1e-3  # learning rate for policy
alpha_v = 1e-3   # learning rate for value

pg_agent = PolicyGradient(
    state_dim=state_dim,
    num_actions=num_actions,
    gamma=gamma,
    alpha_pi=alpha_pi,
    alpha_v=alpha_v,
    device=device,
    policy='mlp'
)

# Training parameters
num_episodes = 5

# Tracking metrics
episode_rewards = []
test_accuracies = []
policy_losses = []
value_losses = []

# Training loop
for episode in range(num_episodes):
    # Episode data collection
    sampled_states = []
    sampled_actions = []
    sampled_rewards = []
    pseudo_labeled_data = []
    
    state, true_label = env.reset()
    total_reward = 0.0
    
    # Collect episode data
    while not env.done:
        # Get action from policy
        action = pg_agent.policy(state)
        
        # Step in environment
        next_state, reward, terminated, truncated = env.step(action if action < 10 else -1)
        
        # Store transition
        sampled_states.append(state)
        sampled_actions.append(action)
        sampled_rewards.append(reward)
        total_reward += reward
        
        # Store pseudo-labeled data
        if action < 10 and env.current_sample is not None:
            x_tensor, _, _ = env.current_sample
            pseudo_labeled_data.append((x_tensor.detach().cpu(), action))
        
        # Move to next state
        if next_state is None:
            break
        state = next_state
    
    # Update policy after episode
    if len(sampled_states) > 0:
        policy_loss, value_loss = pg_agent.update(
            sampled_states, 
            sampled_actions, 
            sampled_rewards
        )
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
    
    # Log episode results
    print(f"Episode {episode + 1}/{num_episodes}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Pseudo-labeled samples: {len(pseudo_labeled_data)}")
    if len(sampled_states) > 0:
        print(f"  Policy Loss: {policy_loss:.4f}")
        print(f"  Value Loss: {value_loss:.4f}")
    
    episode_rewards.append(total_reward)
    
    # Retrain downstream model
    if pseudo_labeled_data:
        model = retrain_downstream_model(model, pseudo_labeled_data, device=device, epochs=3)
        test_acc = evaluate_model_accuracy(model, device)
        test_accuracies.append(test_acc)
        print(f"  Test Accuracy: {test_acc:.4f}")
    
    print("-" * 50)

env.close()

# Save the trained policy
pg_agent.save_model(name='pseudo_labeling_pg', save_dir='src/trained_models')

print("\nTraining completed!")
print(f"Final test accuracy: {test_accuracies[-1]:.4f}" if test_accuracies else "No test accuracy available")

