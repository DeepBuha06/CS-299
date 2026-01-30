"""
Configuration file for Attention-based Text Classification.
Contains all hyperparameters and settings.
"""

import torch
from pathlib import Path


class Config:
    """Configuration class with all hyperparameters."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw" / "imdb"
    VOCAB_FILE = PROJECT_ROOT / "vocab.json"
    MODEL_DIR = PROJECT_ROOT / "checkpoints"
    
    # Data settings
    MAX_SEQ_LENGTH = 256          # Maximum sequence length (truncate/pad to this)
    MIN_WORD_FREQ = 2             # Minimum word frequency to include in vocab
    
    # Vocabulary
    VOCAB_SIZE = 15000            # Maximum vocabulary size
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1
    
    # Embedding settings
    EMBEDDING_DIM = 300           # Dimension of word embeddings
    USE_PRETRAINED = False        # Whether to use pre-trained embeddings (GloVe)
    GLOVE_PATH = None             # Path to GloVe file (if USE_PRETRAINED=True)
    FREEZE_EMBEDDINGS = False     # Whether to freeze embeddings during training
    
    # Encoder settings (BiLSTM)
    HIDDEN_DIM = 256              # LSTM hidden dimension
    NUM_LAYERS = 1                # Number of LSTM layers
    BIDIRECTIONAL = True          # Use bidirectional LSTM
    ENCODER_DROPOUT = 0.3         # Dropout in encoder
    
    # Attention settings
    ATTENTION_TYPE = "additive"   # "additive" (Bahdanau) or "dot"
    ATTENTION_DIM = 128           # Attention hidden dimension (for additive)
    
    # Classifier settings
    CLASSIFIER_DROPOUT = 0.5      # Dropout before classification layer
    NUM_CLASSES = 1               # 1 for binary (sigmoid), >1 for multi-class (softmax)
    
    # Training settings
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 1
    CLIP_GRAD = 1.0               # Gradient clipping
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    SEED = 42
    
    @classmethod
    def get_encoder_output_dim(cls):
        """Get the output dimension of the encoder."""
        multiplier = 2 if cls.BIDIRECTIONAL else 1
        return cls.HIDDEN_DIM * multiplier
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        for key, value in vars(cls).items():
            if not key.startswith("_") and not callable(getattr(cls, key)):
                print(f"  {key}: {value}")
        print("=" * 60)


# Create model directory if it doesn't exist
Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
