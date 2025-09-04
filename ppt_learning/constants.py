"""
Constants used throughout the PPT learning codebase.
This module centralizes magic numbers and configuration values for better maintainability.
"""

# Model Architecture Constants
DEFAULT_EMBED_DIM = 1024
DEFAULT_NUM_BLOCKS = 24
DEFAULT_NUM_HEADS = 16

# Point Cloud Processing Constants
DEFAULT_PCD_POINTS = 1200
DEFAULT_PCD_POINTS_LARGE = 8192
DEFAULT_PCD_CHANNELS = 4

# Vision Model Constants
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224

# Linear Layer Dimensions (policy_stem.py specific)
DINOV2_FEATURE_DIM = 168960  # DINOv2 feature dimension for linear layer

# Training Constants
DEFAULT_MAX_TIMESTEPS = 1300
DEFAULT_ACTION_HORIZON = 1
DEFAULT_OBSERVATION_HORIZON = 4

# Loss and Optimization Constants
DEFAULT_K_VALUE = 0.01  # Exponential weight decay constant in merge_act

# Logging Constants
DEFAULT_LOG_MAXLEN = 50
DEFAULT_PROGRESS_LOG_MAXLEN = 20

# File and I/O Constants
TOKENIZERS_PARALLELISM = "false"

# Import torch for device check
import torch

# Device Constants
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
