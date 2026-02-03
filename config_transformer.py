"""
Configuration file for Transformer-based Text Classification.
Uses DistilBERT for sentiment analysis.
"""

import torch
from pathlib import Path


class TransformerConfig:
    """Configuration class for transformer model hyperparameters."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw" / "imdb"
    MODEL_DIR = PROJECT_ROOT / "checkpoints"
    
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"  # HuggingFace model name
    NUM_LABELS = 2  # Binary classification (positive/negative)
    
    # Data settings
    MAX_SEQ_LENGTH = 512  # Maximum sequence length for transformer
    
    # Training settings
    BATCH_SIZE = 16  # Smaller batch size due to memory constraints
    LEARNING_RATE = 2e-5  # Standard learning rate for fine-tuning
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 3  # Transformers converge faster
    WARMUP_RATIO = 0.1  # Warmup for 10% of training steps
    
    # Gradient settings
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32
    MAX_GRAD_NORM = 1.0
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    SEED = 42
    
    # Checkpoint names
    MODEL_CHECKPOINT = "transformer_model.pt"
    METRICS_FILE = "transformer_metrics.json"
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("=" * 60)
        print("TRANSFORMER CONFIGURATION")
        print("=" * 60)
        for key, value in vars(cls).items():
            if not key.startswith("_") and not callable(getattr(cls, key)):
                print(f"  {key}: {value}")
        print("=" * 60)


# Create model directory if it doesn't exist
TransformerConfig.MODEL_DIR.mkdir(parents=True, exist_ok=True)
