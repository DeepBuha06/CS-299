"""
Configuration for Experiment 3: Comprehensiveness Test
"""

import torch
from pathlib import Path

class ExperimentConfig:
    """Configuration for comprehensiveness experiment."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "bilstm_model.pt"
    VOCAB_FILE = PROJECT_ROOT / "vocab.json"
    
    # Experiment settings
    TOP_K_VALUES = [1, 5, 10]  # Different k values to test
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model config (must match the trained model)
    VOCAB_SIZE = 15002
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    ATTENTION_DIM = 128
    NUM_LAYERS = 1
    BIDIRECTIONAL = True
    ATTENTION_TYPE = "additive"
    ENCODER_DROPOUT = 0.3
    CLASSIFIER_DROPOUT = 0.5
    NUM_CLASSES = 1
    PAD_IDX = 0
    UNK_IDX = 1
    
    # Preprocessing settings
    MAX_SEQ_LENGTH = 256
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
