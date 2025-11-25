"""
Global configuration values for the small-scale Pitot-style project.
"""

RANDOM_SEED = 42

# Synthetic dataset configuration
N_WORKLOADS = 30
N_PLATFORMS = 10
MAX_INTERFERERS = 3
N_SAMPLES = 4000

# Train/test split
TRAIN_FRACTION = 0.8

# Noise level on log-runtime
LOG_NOISE_STD = 0.15  # small; keeps problem learnable

# Model hyperparameters
HIDDEN_UNITS = (64, 64)
MAX_EPOCHS = 250
LEARNING_RATE = 1e-3

# Pruning configuration
PRUNE_FRACTION = 0.5  # prune 50% smallest-magnitude weights

# Results directory
RESULTS_DIR = "results"
