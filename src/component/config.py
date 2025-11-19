# Central config for hyperparameters

# Training parameters
episodes = 50
num_labeled = 1000
num_unlabeled = 4000
num_validation = 1000

# Policy Gradient parameters
gamma = 0.99
alpha_pi = 1e-3
alpha_v = 1e-3
hidden_dim = 64
policy_type = 'mlp'

# Downstream Model parameters
downstream_lr = 0.001
downstream_epocs = 1
batch_size = 32