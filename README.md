# Attention is NOT Explanation: Implementation & Experiments

Implementation of the paper **"Attention is not Explanation"** by Jain & Wallace (2019) to empirically test whether attention weights provide faithful explanations of model predictions.

## 📄 Research Question

**Can attention weights be trusted as explanations for model predictions?**

This project implements experiments to test the claim that attention mechanisms, while useful for model performance, do not necessarily provide reliable explanations of how neural networks make decisions.

## 🎯 Project Status

### 📊 Dataset & Task
- **Dataset**: IMDB Movie Reviews (50,000 reviews)
- **Task**: Binary Sentiment Classification (Positive/Negative)
- **Split**: 25,000 training, 25,000 testing
- **Domain**: Movie review text analysis

### ✅ Completed (Phase 1: Model Setup)
- BiLSTM with Attention mechanism (86.2% accuracy on IMDB)
- Transformer (DistilBERT) fine-tuned (90.8% accuracy on IMDB)
- Attention weight extraction for both architectures
- Web interface for attention visualization
- Baseline performance established on sentiment classification

### 🔬 In Progress (Phase 2: Experiments from Paper)
The following experiments from the paper are yet to be implemented:

1. **Attention vs Gradient Correlation** - Test if attention correlates with gradient-based feature importance
2. **Adversarial Attention** - Create different attention distributions that yield same predictions
3. **Permutation Tests** - Verify if high-attention words are truly important
4. **Counterfactual Analysis** - Test causal relationship between attention and predictions
5. **Attention Weight Randomization** - Compare model behavior with shuffled attention

## 🏗️ Project Structure

```
CS-299/
├── webapp/
│   ├── app.py              # Flask application
│   ├── static/
│   │   ├── script.js       # Frontend JavaScript
│   │   └── style.css       # Styles
│   └── templates/
│       └── index.html      # Main page
├── models/                 # BiLSTM model architecture
├── models_transformer/     # Transformer model architecture
├── checkpoints/
│   ├── bilstm_model.pt     # Trained BiLSTM weights
│   ├── transformer_model.pt # Trained Transformer weights
│   ├── bilstm_metrics.json
│   └── transformer_metrics.json
├── vocab.json              # Vocabulary for BiLSTM
├── requirements.txt        # Python dependencies
├── Procfile               # Deployment configuration
└── render.yaml            # Render deployment config
```

## 📊 Model Performance (IMDB Sentiment Analysis)

Both models trained on binary sentiment classification (Positive/Negative reviews):

### BiLSTM + Attention
- **Accuracy**: 86.2%
- **F1 Score**: 85.4%
- **Precision**: 90.7%
- **Recall**: 80.7%
- **Architecture**: Bidirectional LSTM with Additive Attention

### Transformer (DistilBERT)
- **Accuracy**: 90.8%
- **F1 Score**: 90.9%
- **Precision**: 89.9%
- **Recall**: 91.8%
- **Architecture**: Fine-tuned DistilBERT

**Dataset**: IMDB Movie Reviews (25,000 train / 25,000 test)

## 🔬 Research Methodology

### Paper Reference
**"Attention is not Explanation"** - Sarthak Jain & Byron C. Wallace (2019)
[Paper Link](https://arxiv.org/abs/1902.10186)

### Key Claims to Test
1. **Attention weights do not correlate with gradient-based measures** of feature importance
2. **Adversarial attention distributions exist** - different attention can yield same predictions
3. **High attention ≠ high importance** - removing high-attention words may not change predictions

### Experimental Pipeline
```
Phase 1: Setup ✅
├── Build attention-based models (BiLSTM + Transformer)
├── Train on IMDB sentiment dataset
├── Extract attention weights
└── CrCurrent Usage (Baseline Models)

The webapp currently provides:
1. **Model Selection**: BiLSTM + Attention or DistilBERT Transformer
2. **Sentiment Prediction**: Binary classification (Positive/Negative)
3. **Attention Visualization**: View attention weights (not necessarily explanations!)
4. **Performance Metrics**: Accuracy, F1, Precision, Recall

**Note**: The attention visualizations show what the model *attends to*, but experiments will test whether this actually *explains* the predictions.ming)
├── Quantitative results comparison
├── Visualization of findings
├── Documentation of limitations
└── Conclusion validation
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/Ramji-Purwar/CS-299.git
cd CS-299
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python webapp/app.py
```

4. **Open in browser**
```
http://localhost:5000
```

## 🙏 Acknowledgments

- IMDB Dataset
- HuggingFace Transformers
- "Attention is not Explanation" paper authors

---
