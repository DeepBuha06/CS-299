import torch
from pathlib import Path


class TransformerConfig:
    
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw" / "imdb"
    MODEL_DIR = PROJECT_ROOT / "checkpoints"
    
    MODEL_NAME = "distilbert-base-uncased"  # HuggingFace model name
    NUM_LABELS = 2  # Binary classification (positive/negative)
    
    MAX_SEQ_LENGTH = 256  # Maximum sequence length for transformer
    
    BATCH_SIZE = 16  # Smaller batch size due to memory constraints
    LEARNING_RATE = 2e-5  # Standard learning rate for fine-tuning
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 1  # Single epoch for quick training
    WARMUP_RATIO = 0.1  # Warmup for 10% of training steps
    
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32
    MAX_GRAD_NORM = 1.0
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    SEED = 42
    
    MODEL_CHECKPOINT = "transformer_model.pt"
    METRICS_FILE = "transformer_metrics.json"
    
    @classmethod
    def print_config(cls):
        for key, value in vars(cls).items():
            if not key.startswith("_") and not callable(getattr(cls, key)):
                print(f"  {key}: {value}")


TransformerConfig.MODEL_DIR.mkdir(parents=True, exist_ok=True)
