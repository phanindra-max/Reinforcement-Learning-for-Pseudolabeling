# Central config for hyperparameters

# Training parameters
EPISODES = 50
NUM_LABELED = 1000
NUM_UNLABELED = 4000
NUM_VALIDATION = 1000

# Policy Gradient parameters
GAMMA = 0.99
ALPHA_PI = 1e-3
ALPHA_V = 1e-3
HIDDEN_DIM = 64
POLICY_TYPE = 'mlp'

# Downstream Model parameters
DOWNSTREAM_LR = 0.001
DOWNSTREAM_EPOCHS = 1
BATCH_SIZE = 32