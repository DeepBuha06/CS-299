# 🎬 Sentiment Analysis - Attention Visualizer

Live sentiment analysis web application comparing BiLSTM with Attention and Transformer (DistilBERT) models for IMDB movie reviews.

## 🌟 Features

- **Dual Model Architecture**: Switch between BiLSTM + Attention and DistilBERT Transformer
- **Real-time Sentiment Analysis**: Instant predictions on movie reviews
- **Attention Visualization**: See which words influenced the model's decision
- **Interactive UI**: Clean, modern interface with responsive design
- **Model Metrics**: View accuracy, F1 score, precision, and recall

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

## 📦 Deployment

### Deploy to Render (Recommended)

1. **Push to GitHub** (if not already done)
```bash
git add .
git commit -m "Add deployment configuration"
git push origin main
```

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Deploy**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `Ramji-Purwar/CS-299`
   - Render will automatically detect `render.yaml`
   - Click "Create Web Service"

4. **Wait for deployment** (5-10 minutes)
   - Models will be loaded automatically
   - You'll get a live URL like: `https://sentiment-analysis-webapp.onrender.com`

### Deploy to Heroku

```bash
# Install Heroku CLI
heroku login
heroku create sentiment-analysis-app

# Deploy
git push heroku main

# Open app
heroku open
```

### Deploy to Railway

1. Go to [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub repo"
3. Select `Ramji-Purwar/CS-299`
4. Railway auto-detects and deploys

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

## 📊 Model Performance

### BiLSTM + Attention
- **Accuracy**: 86.2%
- **F1 Score**: 85.4%
- **Architecture**: Bidirectional LSTM with Additive Attention

### Transformer (DistilBERT)
- **Accuracy**: 90.8%
- **F1 Score**: 90.9%
- **Architecture**: Fine-tuned DistilBERT

## 🔬 Research Context

This project implements models from the paper:
**"Attention is not Explanation"** - Jain & Wallace (2019)

The webapp demonstrates:
- How attention mechanisms work in neural networks
- Visual interpretation of model decisions
- Comparison between different architectures

## 🛠️ Technologies

- **Backend**: Flask, PyTorch
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Models**: BiLSTM, DistilBERT (HuggingFace Transformers)
- **Dataset**: IMDB Movie Reviews (50k reviews)

## 📝 Usage

1. **Select Model**: Choose between BiLSTM or Transformer
2. **Enter Review**: Type or paste a movie review
3. **Analyze**: Click the analyze button
4. **View Results**:
   - Sentiment prediction (Positive/Negative)
   - Confidence score
   - Attention weights visualization
   - Top influential words

## ⚙️ Configuration

Edit model parameters in:
- `config.py` - BiLSTM configuration
- `config_transformer.py` - Transformer configuration

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 👤 Author

**Ramji Purwar**
- GitHub: [@Ramji-Purwar](https://github.com/Ramji-Purwar)
- Repository: [CS-299](https://github.com/Ramji-Purwar/CS-299)

## 🙏 Acknowledgments

- IMDB Dataset
- HuggingFace Transformers
- "Attention is not Explanation" paper authors

---

⭐ Star this repo if you find it helpful!
