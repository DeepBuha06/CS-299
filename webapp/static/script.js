// Sentiment Analysis Web App - JavaScript

document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const reviewInput = document.getElementById('reviewInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const charCount = document.getElementById('charCount');
    const resultSection = document.getElementById('resultSection');
    const sentimentCard = document.getElementById('sentimentCard');
    const sentimentLabel = document.getElementById('sentimentLabel');
    const confidenceValue = document.getElementById('confidenceValue');

    // Character counter
    reviewInput.addEventListener('input', function () {
        charCount.textContent = this.value.length;
    });

    // Analyze button click
    analyzeBtn.addEventListener('click', analyzeSentiment);

    // Enter key to analyze (Ctrl/Cmd + Enter)
    reviewInput.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            analyzeSentiment();
        }
    });

    // Main analysis function
    async function analyzeSentiment() {
        const text = reviewInput.value.trim();

        if (!text) {
            showError('Please enter a review to analyze');
            return;
        }

        // Show loading state
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ review: text })
            });

            const data = await response.json();

            if (response.ok) {
                displayResults(data);
            } else {
                showError(data.error || 'An error occurred');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Failed to connect to server');
        } finally {
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
        }
    }

    // Display results
    function displayResults(data) {
        // Show result section
        resultSection.classList.remove('hidden');

        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Update sentiment card class
        sentimentCard.className = 'card result-card ' + data.sentiment;

        // Update label
        if (data.sentiment === 'positive') {
            sentimentLabel.textContent = 'Positive';
        } else {
            sentimentLabel.textContent = 'Negative';
        }

        // Update probability and confidence
        document.getElementById('probabilityValue').textContent = data.probability + '%';
        confidenceValue.textContent = data.confidence + '%';

        // Render attention visualization
        renderAttention(data.attention, data.sentiment);
    }

    // Global chart variable to store the chart instance
    let attentionChart = null;

    // Render attention weights
    function renderAttention(attentionData, sentiment) {
        if (!attentionData || attentionData.length === 0) {
            return;
        }

        // Create bar chart
        createAttentionChart(attentionData, sentiment);
    }

    // Create attention weights bar chart
    function createAttentionChart(attentionData, sentiment) {
        const ctx = document.getElementById('attentionChart');

        // Destroy previous chart if it exists
        if (attentionChart) {
            attentionChart.destroy();
        }

        // Limit to top 20 words for better readability
        const topWords = attentionData
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 20);

        const words = topWords.map(d => d.word);
        const weights = topWords.map(d => d.weight);

        // Set color based on sentiment
        const barColor = sentiment === 'positive' ? 'rgba(76, 175, 80, 0.8)' : 'rgba(244, 67, 54, 0.8)';
        const borderColor = sentiment === 'positive' ? 'rgba(76, 175, 80, 1)' : 'rgba(244, 67, 54, 1)';

        attentionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: words,
                datasets: [{
                    label: 'Attention Weight',
                    data: weights,
                    backgroundColor: barColor,
                    borderColor: borderColor,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Top 20 Words by Attention Weight',
                        font: {
                            size: 14
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return 'Weight: ' + (context.parsed.y * 100).toFixed(2) + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Words'
                        },
                        ticks: {
                            maxRotation: 90,
                            minRotation: 45,
                            font: {
                                size: 10
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Attention Weight'
                        },
                        beginAtZero: true,
                        ticks: {
                            callback: function (value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // Show error message
    function showError(message) {
        // Create error toast
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.innerHTML = `
            <span class="error-icon">⚠️</span>
            <span class="error-message">${message}</span>
        `;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 90, 90, 0.9);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            z-index: 1000;
            animation: slideIn 0.3s ease;
            backdrop-filter: blur(10px);
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // Load model metrics
    async function loadMetrics() {
        try {
            const response = await fetch('/metrics');
            const data = await response.json();

            if (data.test_accuracy) {
                document.getElementById('modelAccuracy').textContent =
                    (data.test_accuracy * 100).toFixed(1) + '%';
            }
            if (data.test_f1) {
                document.getElementById('modelF1').textContent =
                    data.test_f1.toFixed(3);
            }
        } catch (error) {
            console.log('Could not load metrics');
        }
    }

    // Initialize
    loadMetrics();

    // Add animation styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
});
