
import torch
import json
from pathlib import Path

from config import ExperimentConfig
from comprehensiveness import ComprehensivenessTester
from sufficiency import SufficiencyTester

import sys
sys.path.insert(0, str(ExperimentConfig.PROJECT_ROOT))

from models.model import AttentionClassifier
from data.preprocessing import Preprocessor


def load_model_and_preprocessor():
    preprocessor = Preprocessor.from_vocab_file(
        vocab_path=str(ExperimentConfig.VOCAB_FILE),
        max_length=ExperimentConfig.MAX_SEQ_LENGTH
    )
    
    model = AttentionClassifier(
        vocab_size=ExperimentConfig.VOCAB_SIZE,
        embedding_dim=ExperimentConfig.EMBEDDING_DIM,
        hidden_dim=ExperimentConfig.HIDDEN_DIM,
        attention_dim=ExperimentConfig.ATTENTION_DIM,
        num_classes=ExperimentConfig.NUM_CLASSES,
        num_layers=ExperimentConfig.NUM_LAYERS,
        bidirectional=ExperimentConfig.BIDIRECTIONAL,
        attention_type=ExperimentConfig.ATTENTION_TYPE,
        encoder_dropout=ExperimentConfig.ENCODER_DROPOUT,
        classifier_dropout=ExperimentConfig.CLASSIFIER_DROPOUT,
        padding_idx=ExperimentConfig.PAD_IDX
    )
    
    checkpoint = torch.load(
        str(ExperimentConfig.CHECKPOINT_PATH),
        map_location=ExperimentConfig.DEVICE
    )
    model.load_state_dict(checkpoint)
    model = model.to(ExperimentConfig.DEVICE)
    model.eval()
    
    return model, preprocessor


def run_comprehensiveness_test(review_text: str, top_k_values=None):
    if top_k_values is None:
        top_k_values = ExperimentConfig.TOP_K_VALUES
    
    model, preprocessor = load_model_and_preprocessor()
    
    tester = ComprehensivenessTester(model, preprocessor)
    
    results = tester.compute_multiple_k(review_text, k_values=top_k_values)
    
    return results


def display_detailed_analysis(results):
    pass


def run_sufficiency_test(review_text: str, top_k_values=None):
    if top_k_values is None:
        top_k_values = ExperimentConfig.TOP_K_VALUES
    
    model, preprocessor = load_model_and_preprocessor()
    
    tester = SufficiencyTester(model, preprocessor)
    
    results = tester.compute_multiple_k(review_text, k_values=top_k_values)
    
    return results


def display_sufficiency_analysis(results):
    pass


if __name__ == "__main__":
    sample_review = """
    This movie was absolutely fantastic! The acting was superb, the plot was
    engaging and kept me on the edge of my seat the whole time. The cinematography
    was beautiful and the soundtrack was perfectly suited to the scenes. I would
    highly recommend this movie to anyone who enjoys a good drama. It's one of the
    best films I've seen all year.
    """
    
    comprehensiveness_results = run_comprehensiveness_test(sample_review)
    
    sufficiency_results = run_sufficiency_test(sample_review)
    
    output_file = Path(__file__).parent / "sample_result.json"
    with open(output_file, 'w') as f:
        combined_results = {
            "original_text": sample_review.strip(),
            "comprehensiveness": {
                "results_by_k": {
                    str(k): {
                        "top_k": v["top_k"],
                        "original_prediction": float(v["original_prediction"]),
                        "modified_prediction": float(v["modified_prediction"]),
                        "comprehensiveness": float(v["comprehensiveness"]),
                        "removed_tokens": v["removed_tokens"],
                        "removed_indices": v["removed_indices"],
                        "removed_scores": [float(s) for s in v["removed_scores"]]
                    }
                    for k, v in comprehensiveness_results["results_by_k"].items()
                }
            },
            "sufficiency": {
                "results_by_k": {
                    str(k): {
                        "k": v["k"],
                        "original_prediction": float(v["original_prediction"]),
                        "rationale_prediction": float(v["rationale_prediction"]),
                        "sufficiency": float(v["sufficiency"]),
                        "rationale_tokens": v["rationale_tokens"],
                        "kept_indices": v["kept_indices"],
                        "rationale_scores": [float(s) for s in v["rationale_scores"]]
                    }
                    for k, v in sufficiency_results["results_by_k"].items()
                }
            }
        }
        json.dump(combined_results, f, indent=2)
