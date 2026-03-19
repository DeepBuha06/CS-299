"""
Generate presentation-ready visualizations from adversarial attention experiment results.

Reads the saved results from run_full_test.py and produces PNG plots + a text report.

Usage:
    cd c:\project\CS-299-main
    python experiment_2/generate_plots.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "full_test_results.json"

# Use a clean style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'figure.titlesize': 16,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
})

COLORS = {
    'primary': '#2563eb',
    'secondary': '#dc2626',
    'accent': '#16a34a',
    'entropy': '#8b5cf6',
    'permutation': '#f59e0b',
    'random': '#06b6d4',
    'positive': '#22c55e',
    'negative': '#ef4444',
}


def load_results():
    """Load the experiment results JSON."""
    print(f"Loading results from: {RESULTS_FILE}")
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    results = data['results']
    print(f"Loaded {len(results)} sample results.")
    return results, data


def plot_l1_histogram(results):
    """Plot 1: Distribution of L1 attention differences."""
    l1_diffs = [r['l1_difference'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(l1_diffs, bins=50, color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axvline(np.mean(l1_diffs), color=COLORS['secondary'], linewidth=2, linestyle='--',
               label=f'Mean = {np.mean(l1_diffs):.3f}')
    ax.set_xlabel('L1 Attention Difference (sum of |original - adversarial|)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Adversarial Attention Difference (L1)')
    ax.legend(fontsize=12)

    plt.tight_layout()
    path = RESULTS_DIR / 'plot_l1_histogram.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_prediction_scatter(results):
    """Plot 2: Original vs Adversarial Prediction scatter plot."""
    orig_preds = [r['original_prediction'] for r in results]
    adv_preds = [r['adversarial_prediction'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(orig_preds, adv_preds, alpha=0.15, s=10, c=COLORS['primary'], rasterized=True)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y = x (identical prediction)')
    ax.set_xlabel('Original Model Prediction')
    ax.set_ylabel('Adversarial Prediction (with altered attention)')
    ax.set_title('Original vs Adversarial Prediction\n(Points on line = same prediction despite different attention)')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    plt.tight_layout()
    path = RESULTS_DIR / 'plot_prediction_scatter.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_difference_by_class(results):
    """Plot 3: Box plot of attention differences by true sentiment class."""
    pos_diffs = [r['l1_difference'] for r in results if r['true_label'] == 1]
    neg_diffs = [r['l1_difference'] for r in results if r['true_label'] == 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        [neg_diffs, pos_diffs],
        labels=['Negative Reviews', 'Positive Reviews'],
        patch_artist=True,
        widths=0.5,
        showfliers=False  # hide outliers for cleaner look
    )
    bp['boxes'][0].set_facecolor(COLORS['negative'])
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(COLORS['positive'])
    bp['boxes'][1].set_alpha(0.6)

    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    ax.set_ylabel('L1 Attention Difference')
    ax.set_title('Adversarial Attention Difference by Sentiment Class')

    # Add count labels
    ax.text(1, ax.get_ylim()[1] * 0.95, f'n = {len(neg_diffs)}', ha='center', fontsize=11, style='italic')
    ax.text(2, ax.get_ylim()[1] * 0.95, f'n = {len(pos_diffs)}', ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    path = RESULTS_DIR / 'plot_diff_by_class.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_best_method_bar(results):
    """Plot 4: Distribution of which attack method was best."""
    method_counts = {}
    for r in results:
        m = r['best_method']
        method_counts[m] = method_counts.get(m, 0) + 1

    methods = sorted(method_counts.keys())
    counts = [method_counts[m] for m in methods]
    colors = [COLORS.get(m, COLORS['primary']) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, counts, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        pct = 100 * count / len(results)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(results)*0.01,
                f'{count}\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Adversarial Attack Method')
    ax.set_ylabel('Number of Samples (best method)')
    ax.set_title('Which Attack Method Produced Highest Attention Difference?')

    plt.tight_layout()
    path = RESULTS_DIR / 'plot_best_method.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cosine_similarity_histogram(results):
    """Plot 5: Histogram of cosine similarity between original and adversarial attention."""
    cos_sims = [r['cosine_similarity'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cos_sims, bins=50, color=COLORS['accent'], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axvline(np.mean(cos_sims), color=COLORS['secondary'], linewidth=2, linestyle='--',
               label=f'Mean = {np.mean(cos_sims):.3f}')
    ax.set_xlabel('Cosine Similarity (original vs adversarial attention)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Cosine Similarity Between Original and Adversarial Attention')
    ax.legend(fontsize=12)

    plt.tight_layout()
    path = RESULTS_DIR / 'plot_cosine_similarity.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_same_class_pie(results):
    """Plot 6: Pie chart showing % of samples where prediction class stayed the same."""
    same = sum(1 for r in results if r['same_class'])
    changed = len(results) - same

    fig, ax = plt.subplots(figsize=(7, 7))
    sizes = [same, changed]
    labels = [f'Same Class\n({same})', f'Changed Class\n({changed})']
    colors = [COLORS['positive'], COLORS['negative']]
    explode = (0.03, 0.03)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 13}
    )
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')

    ax.set_title('Did the Prediction Class Change\nDespite Different Attention?', fontsize=15, fontweight='bold')

    plt.tight_layout()
    path = RESULTS_DIR / 'plot_same_class_pie.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_prediction_diff_histogram(results):
    """Plot 7: Histogram of prediction difference (should be near 0)."""
    pred_diffs = [r['prediction_difference'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pred_diffs, bins=50, color=COLORS['secondary'], alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(np.mean(pred_diffs), color='black', linewidth=2, linestyle='--',
               label=f'Mean = {np.mean(pred_diffs):.4f}')
    ax.set_xlabel('|Original Prediction - Adversarial Prediction|')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Prediction Difference\n(Lower = attention change did NOT affect prediction)')
    ax.legend(fontsize=12)

    plt.tight_layout()
    path = RESULTS_DIR / 'plot_prediction_diff.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_top5_overlap_bar(results):
    """Plot 8: Distribution of top-5 word overlap counts."""
    overlaps = [r['top5_overlap'] for r in results]

    counts = [overlaps.count(i) for i in range(6)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(6), counts, color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=1.5)

    for bar, count in zip(bars, counts):
        pct = 100 * count / len(results)
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(results)*0.005,
                    f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Number of Shared Words in Top-5 (out of 5)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Top-5 Attended Word Overlap:\nOriginal vs Adversarial Attention')
    ax.set_xticks(range(6))

    plt.tight_layout()
    path = RESULTS_DIR / 'plot_top5_overlap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def create_summary_dashboard(results, data):
    """Create a combined summary figure with key stats."""
    l1_diffs = [r['l1_difference'] for r in results]
    cos_sims = [r['cosine_similarity'] for r in results]
    pred_diffs = [r['prediction_difference'] for r in results]
    same_count = sum(1 for r in results if r['same_class'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment 2: Adversarial Attention Attack — Summary Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)

    # Top-left: L1 histogram
    ax = axes[0, 0]
    ax.hist(l1_diffs, bins=40, color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(l1_diffs), color=COLORS['secondary'], linewidth=2, linestyle='--',
               label=f'Mean = {np.mean(l1_diffs):.3f}')
    ax.set_xlabel('L1 Attention Difference')
    ax.set_ylabel('Count')
    ax.set_title('Attention Difference Distribution')
    ax.legend()

    # Top-right: Prediction scatter
    ax = axes[0, 1]
    orig = [r['original_prediction'] for r in results]
    adv = [r['adversarial_prediction'] for r in results]
    ax.scatter(orig, adv, alpha=0.1, s=5, c=COLORS['primary'], rasterized=True)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
    ax.set_xlabel('Original Prediction')
    ax.set_ylabel('Adversarial Prediction')
    ax.set_title('Prediction Stability')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    # Bottom-left: Same Class pie
    ax = axes[1, 0]
    same = same_count
    changed = len(results) - same
    ax.pie([same, changed],
           labels=[f'Same ({same})', f'Changed ({changed})'],
           colors=[COLORS['positive'], COLORS['negative']],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    ax.set_title('Prediction Class Stability')

    # Bottom-right: Key stats text
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = (
        f"Total Samples:   {len(results):,}\n"
        f"Time:            {data.get('total_time_seconds', 0)/60:.1f} min\n"
        f"\n"
        f"Avg L1 Diff:     {np.mean(l1_diffs):.4f} ± {np.std(l1_diffs):.4f}\n"
        f"Avg Cosine Sim:  {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f}\n"
        f"Avg Pred Diff:   {np.mean(pred_diffs):.4f} ± {np.std(pred_diffs):.4f}\n"
        f"\n"
        f"Same Class Rate: {100*same_count/len(results):.1f}%\n"
        f"\n"
        f"CONCLUSION:\n"
        f"Attention is NOT a faithful explanation.\n"
        f"Different attention → Same prediction."
    )
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f4f8', edgecolor='#cbd5e1'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = RESULTS_DIR / 'plot_summary_dashboard.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    if not RESULTS_FILE.exists():
        print(f"ERROR: Results file not found at {RESULTS_FILE}")
        print("Run 'python experiment_2/run_full_test.py' first!")
        return

    results, data = load_results()

    print(f"\nGenerating visualizations...")
    print(f"Output directory: {RESULTS_DIR}\n")

    # Generate all plots
    plot_l1_histogram(results)
    plot_prediction_scatter(results)
    plot_difference_by_class(results)
    plot_best_method_bar(results)
    plot_cosine_similarity_histogram(results)
    plot_same_class_pie(results)
    plot_prediction_diff_histogram(results)
    plot_top5_overlap_bar(results)
    create_summary_dashboard(results, data)

    print(f"\n{'='*70}")
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nAll PNG files are in: {RESULTS_DIR}")
    print("\nPlots created:")
    for png_file in sorted(RESULTS_DIR.glob('plot_*.png')):
        print(f"  - {png_file.name}")
    print("\nThese are ready to use in your presentation/report!")


if __name__ == '__main__':
    main()
