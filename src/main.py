import torch
import numpy as np
from src.model import SimpleCNN, DQN, PolicyGradient
from src.env import PseudoLabelEnv
from component.utils import retrain_downstream_model, evaluate_model, ReplayBuffer, get_ssl_mnist, evaluate_model_accuracy
import matplotlib.pyplot as plt

'''
1. Load labeled and unlabeled data
2. Init CNN and DQN models
3. For each episode:
   - Use DQN to decide pseudo-labels
   - Get reward (if action is valid)
   - Store transitions in replay buffer
   - Sample mini-batches to train:
     - DQN (TD loss)
     - CNN (supervised CE loss)
4. Evaluate model
5. Plot metrics at the end
'''

# Initialization
# ------------------------------------

device = torch.device("cpu")
# Loading labeled and unlabeled data
labeled_loader, unlabeled_loader = get_ssl_mnist(labeled_size=100, unlabeled_size=2000)

# Model
model = SimpleCNN().to(device)
# Initializing custom environment
env = PseudoLabelEnv(unlabeled_loader, model, device)

# DQN
input_dim = 11 # probs (10) + entropy (1)
output_dim = 11 # 10 labels + 1 skip
batch_size = 32
gamma = 0.99 # discount factor for reward
epsilon = 0.1 # exploration rate to balance both exploring the environment and exploiting known information
epsilon_min = 0.01 # min value for epsilon
epsilon_decay = 0.995 # rate to reduce epsilon value
num_episodes = 5

policy_net = DQN(input_dim, output_dim).to(device) # learning Q-values for labeling decisions
target_net = DQN(input_dim, output_dim).to(device) # stabilizing Q-learning
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(capacity=1000)

# Tracking lists for plotting
episode_rewards = []
test_accuracies = []
losses_ce_labeled = []
losses_ce_pseudo = []
losses_dqn = []

# Training Loop
# ------------------------------------
for episode in range(num_episodes): # running the loop for complete pass of unlabeled data each time
    state, true_label = env.reset()
    total_reward = 0.0
    pseudo_labeled_data = []

    while not env.done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(output_dim)
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        # Step in environment
        next_state, reward, terminated, truncated = env.step(action if action < 10 else -1)
        total_reward += reward

        # Store only real labels (0-9)
        if action != -1 and env.current_sample is not None and action < 10:
            x_tensor, _, _ = env.current_sample
            pseudo_labeled_data.append((x_tensor.detach().cpu(), action))

        # Push transition to replay buffer
        done_flag = 1.0 if terminated else 0.0
        if next_state is None:
            next_state = np.zeros_like(state)  # dummy state for terminal
        replay_buffer.push(state, action, reward, next_state, done_flag)

        # DQN + Supervised Loss training step
        if len(replay_buffer) >= batch_size:
            s_batch, a_batch, r_batch, s_next_batch, d_batch = replay_buffer.sample(batch_size)

            # ---- DQN LOSS ----
            q_values = policy_net(s_batch.to(device))
            q_value = q_values.gather(1, a_batch.unsqueeze(1).to(device)).squeeze(1)

            with torch.no_grad():
                q_next = target_net(s_next_batch.to(device)).max(1)[0]
                q_target = r_batch.to(device) + gamma * q_next * (1 - d_batch.to(device))

            dqn_loss = torch.nn.functional.mse_loss(q_value, q_target)

            # ---- Supervised LOSS ----
            import torch.nn as nn

            ce_loss_fn = nn.CrossEntropyLoss()

            # 1. Labeled data (fixed small set)
            try:
                labeled_iter
            except NameError:
                labeled_iter = iter(labeled_loader)

            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)

            x_labeled, y_labeled = labeled_batch
            x_labeled, y_labeled = x_labeled.to(device), y_labeled.to(device)

            logits_labeled = model(x_labeled)
            loss_labeled = ce_loss_fn(logits_labeled, y_labeled)

            # 2. Pseudo-labeled data
            if pseudo_labeled_data:
                import random

                sampled = random.sample(pseudo_labeled_data, min(batch_size, len(pseudo_labeled_data)))
                x_list, y_list = zip(*sampled)
                x_tensor = torch.cat(x_list, dim=0).to(device)
                y_tensor = torch.tensor(y_list).to(device)

                logits_pseudo = model(x_tensor)
                loss_pseudo = ce_loss_fn(logits_pseudo, y_tensor)
            else:
                loss_pseudo = torch.tensor(0.0).to(device)

            losses_ce_labeled.append(loss_labeled.item())
            losses_ce_pseudo.append(loss_pseudo.item() if isinstance(loss_pseudo, torch.Tensor) else 0.0)
            losses_dqn.append(dqn_loss.item())

            # ---- Combine all losses ----
            total_loss = dqn_loss + loss_labeled + loss_pseudo

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # move to next state
        state = next_state

    # end of episode
    print(f"Episode {episode + 1} reward: {total_reward}")
    episode_rewards.append(total_reward)
    print(f"Collected {len(pseudo_labeled_data)} pseudo-labeled samples.")

    # Retrain downstream CNN on pseudo-labeled data
    if pseudo_labeled_data:
        model = retrain_downstream_model(model, pseudo_labeled_data, device=device, epochs=3)
        evaluate_model(model, device)
        test_acc = evaluate_model_accuracy(model, device)
        test_accuracies.append(test_acc)

    # Update target network & decay epsilon
    target_net.load_state_dict(policy_net.state_dict())
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()

# plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(episode_rewards, label="Reward")
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(test_accuracies, label="Test Accuracy")
plt.title("Test Accuracy")
plt.xlabel("Episode")
plt.ylabel("Accuracy")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(losses_ce_labeled, label="CE (labeled)", alpha=0.7)
plt.plot(losses_ce_pseudo, label="CE (pseudo)", alpha=0.7)
plt.plot(losses_dqn, label="DQN loss", alpha=0.7)
plt.title("Training Losses")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
