# PolicyGradient Usage Guide

## Overview

The `PolicyGradient` class has been successfully converted from TensorFlow to PyTorch and integrated into your project. This implementation uses an Actor-Critic approach with baseline for variance reduction.

## Key Differences: DQN vs PolicyGradient

### DQN (Your Current Approach)
- **Type**: Value-based RL
- **Learning**: Learns Q-values for state-action pairs
- **Action Selection**: ε-greedy (exploration vs exploitation)
- **Updates**: Uses replay buffer with mini-batch sampling
- **Target Network**: Required for stability

### PolicyGradient (New Approach)
- **Type**: Policy-based RL
- **Learning**: Directly learns policy (action probabilities)
- **Action Selection**: Stochastic sampling from learned distribution
- **Updates**: Uses full episode trajectory
- **Value Network**: Used as baseline to reduce variance

## Architecture

### Policy Network (Actor)
```
State (11 dims) → FC(64) → Tanh → FC(64) → Tanh → FC(11) → Logits
```
- **Input**: State vector (10 class probabilities + 1 entropy)
- **Output**: Logits for 11 actions (10 labels + 1 skip)

### Value Network (Critic)
```
State (11 dims) → FC(64) → Tanh → FC(64) → Tanh → FC(1) → Value
```
- **Input**: State vector
- **Output**: Scalar value estimate

## Usage Example

### Basic Initialization
```python
from src.model import PolicyGradient

# Initialize PolicyGradient agent
pg_agent = PolicyGradient(
    state_dim=11,           # 10 probs + 1 entropy
    num_actions=11,         # 10 labels + skip
    gamma=0.99,             # discount factor
    alpha_pi=1e-3,          # policy learning rate
    alpha_v=1e-3,           # value learning rate
    device='cpu',           # or 'cuda'
    policy='mlp'            # multi-layer perceptron
)
```

### Action Selection
```python
# Get action from current policy
action = pg_agent.policy(state)  # Returns integer 0-10
```

### Training Loop (Episodic)
```python
# Collect episode data
sampled_states = []
sampled_actions = []
sampled_rewards = []

# Run episode
state = env.reset()
while not done:
    action = pg_agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    
    sampled_states.append(state)
    sampled_actions.append(action)
    sampled_rewards.append(reward)
    
    state = next_state

# Update after episode
policy_loss, value_loss = pg_agent.update(
    sampled_states, 
    sampled_actions, 
    sampled_rewards
)
```

### Saving and Loading
```python
# Save model
pg_agent.save_model(name='my_model', save_dir='src/trained_models')

# Load model
pg_agent.load_model(name='my_model', save_dir='src/trained_models')
```

## Integration with Your Project

### Replace DQN with PolicyGradient in main.py

**Current DQN approach:**
```python
# Initialize DQN
policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(capacity=1000)

# Action selection
if np.random.rand() < epsilon:
    action = np.random.choice(output_dim)
else:
    with torch.no_grad():
        q_values = policy_net(state_tensor)
        action = q_values.argmax().item()

# Update with replay buffer
if len(replay_buffer) >= batch_size:
    # ... DQN update logic
```

**New PolicyGradient approach:**
```python
# Initialize PolicyGradient
pg_agent = PolicyGradient(
    state_dim=11,
    num_actions=11,
    gamma=0.99,
    alpha_pi=1e-3,
    alpha_v=1e-3,
    device=device,
    policy='mlp'
)

# Collect episode data
episode_states = []
episode_actions = []
episode_rewards = []

# Action selection (no epsilon needed)
action = pg_agent.policy(state)

# Store transitions during episode
episode_states.append(state)
episode_actions.append(action)
episode_rewards.append(reward)

# Update at end of episode
policy_loss, value_loss = pg_agent.update(
    episode_states,
    episode_actions,
    episode_rewards
)
```

## Advantages for Your Project

1. **Direct Policy Learning**: Learns optimal action probabilities directly
2. **Stochastic Policy**: Natural exploration without ε-greedy
3. **Better for Discrete Actions**: Well-suited for classification tasks
4. **Episode-based Learning**: Aligns well with your pseudo-labeling task
5. **Variance Reduction**: Value baseline reduces gradient variance

## Parameters to Tune

- `alpha_pi` (policy learning rate): Start with 1e-3, decrease if unstable
- `alpha_v` (value learning rate): Start with 1e-3
- `gamma` (discount factor): 0.95-0.99 for longer-term rewards
- `hidden_dim`: 64 or 128 for network capacity

## Expected Behavior

- **Early Episodes**: Random-like behavior, low rewards
- **Mid Training**: Policy starts preferring certain actions
- **Late Training**: Consistent high-reward actions

## Troubleshooting

### High Policy Loss
- Decrease `alpha_pi`
- Check reward scaling
- Verify state normalization

### Unstable Training
- Decrease learning rates
- Add gradient clipping
- Normalize advantages

### Low Rewards
- Increase exploration (lower `gamma`)
- Check reward function
- Verify environment logic

## Comparison with Original Code

| Feature | TensorFlow (Original) | PyTorch (Converted) |
|---------|----------------------|---------------------|
| Framework | TensorFlow 2.x + Keras | PyTorch |
| Network Definition | Sequential API | nn.Module classes |
| Action Sampling | tf.random.categorical | Categorical distribution |
| Gradients | GradientTape | .backward() |
| Optimizer | optimizers.Adam | torch.optim.Adam |
| Model Saving | .h5 format | .pth format |
| Device Support | Implicit | Explicit (cpu/cuda) |

## Next Steps

1. Run the example script: `python src/examples/policy_gradient_example.py`
2. Compare performance with your DQN approach
3. Tune hyperparameters for your specific task
4. Consider hybrid approaches if needed

## References

- Original implementation: Your professor's CartPole example
- Algorithm: REINFORCE with baseline (Williams, 1992)
- Framework: PyTorch official documentation

