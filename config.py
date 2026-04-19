import torch
from pathlib import Path


class Config:
    
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw" / "imdb"
    VOCAB_FILE = PROJECT_ROOT / "vocab.json"
    MODEL_DIR = PROJECT_ROOT / "checkpoints"
    
    MAX_SEQ_LENGTH = 256          # Maximum sequence length (truncate/pad to this)
    MIN_WORD_FREQ = 2             # Minimum word frequency to include in vocab
    
    VOCAB_SIZE = 15000            # Maximum vocabulary size
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1
    
    EMBEDDING_DIM = 300           # Dimension of word embeddings
    USE_PRETRAINED = False        # Whether to use pre-trained embeddings (GloVe)
    GLOVE_PATH = None             # Path to GloVe file (if USE_PRETRAINED=True)
    FREEZE_EMBEDDINGS = False     # Whether to freeze embeddings during training
    
    HIDDEN_DIM = 256              # LSTM hidden dimension
    NUM_LAYERS = 1                # Number of LSTM layers
    BIDIRECTIONAL = True          # Use bidirectional LSTM
    ENCODER_DROPOUT = 0.3         # Dropout in encoder
    
    ATTENTION_TYPE = "additive"   # "additive" (Bahdanau) or "dot"
    ATTENTION_DIM = 128           # Attention hidden dimension (for additive)
    
    CLASSIFIER_DROPOUT = 0.5      # Dropout before classification layer
    NUM_CLASSES = 1               # 1 for binary (sigmoid), >1 for multi-class (softmax)
    
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 1
    CLIP_GRAD = 1.0               # Gradient clipping
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    SEED = 42
    
    @classmethod
    def get_encoder_output_dim(cls):
        multiplier = 2 if cls.BIDIRECTIONAL else 1
        return cls.HIDDEN_DIM * multiplier
    
    @classmethod
    def print_config(cls):
        for key, value in vars(cls).items():
            if not key.startswith("_") and not callable(getattr(cls, key)):
                print(f"  {key}: {value}")


Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
