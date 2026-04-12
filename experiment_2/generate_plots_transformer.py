import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results_transformer"
RESULTS_FILE = RESULTS_DIR / "full_test_results_transformer.json"
KENDALL_FILE = RESULTS_DIR / "kendall_tau_transformer.json"

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13,
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.3, 'font.family': 'sans-serif',
})

C = {
    'primary': '#2563eb', 'secondary': '#dc2626', 'accent': '#16a34a',
    'entropy': '#8b5cf6', 'permutation': '#f59e0b', 'random': '#06b6d4',
    'positive': '#22c55e', 'negative': '#ef4444',
}


def load_results():
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    results = data['results']
    return results, data


def plot_l1_histogram(results):
    vals = [r['l1_difference'] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=50, color=C['primary'], alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(vals), color=C['secondary'], lw=2, ls='--', label=f'Mean = {np.mean(vals):.3f}')
    ax.set_xlabel('L1 Attention Difference')
    ax.set_ylabel('Count')
    ax.set_title('Transformer: Distribution of Adversarial Attention Difference (L1)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'plot_l1_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_jsd_histogram(results):
    vals = [r['js_divergence'] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=50, color='#7c3aed', alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(vals), color=C['secondary'], lw=2, ls='--', label=f'Mean JSD = {np.mean(vals):.3f}')
    ax.axvline(np.log(2), color='black', lw=2, ls=':', label=f'Upper Bound (ln2 = {np.log(2):.3f})')
    ax.set_xlabel('Jensen-Shannon Divergence')
    ax.set_ylabel('Count')
    ax.set_title('Transformer: JSD Between Original and Adversarial Attention')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'plot_jsd_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_same_class_pie(results):
    same = sum(1 for r in results if r['same_class'])
    changed = len(results) - same
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        [same, changed], explode=(0.03, 0.03),
        labels=[f'Same Class\n({same})', f'Changed Class\n({changed})'],
        colors=[C['positive'], C['negative']], autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 13})
    for at in autotexts:
        at.set_fontsize(14)
        at.set_fontweight('bold')
    ax.set_title('Transformer: Prediction Stability Despite\nDifferent Attention', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'plot_same_class_pie.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_best_method(results):
    mc = {}
    for r in results:
        m = r['best_method']
        mc[m] = mc.get(m, 0) + 1
    methods = sorted(mc.keys())
    counts = [mc[m] for m in methods]
    colors = [C.get(m, C['primary']) for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, counts, color=colors, alpha=0.85, edgecolor='white', lw=1.5)
    for bar, count in zip(bars, counts):
        pct = 100 * count / len(results)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(results)*0.01,
                f'{count}\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Attack Method')
    ax.set_ylabel('Count')
    ax.set_title('Transformer: Best Attack Method Distribution')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'plot_best_method.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cosine_similarity(results):
    vals = [r['cosine_similarity'] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=50, color=C['accent'], alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(vals), color=C['secondary'], lw=2, ls='--', label=f'Mean = {np.mean(vals):.3f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Transformer: Cosine Similarity (Original vs Adversarial Attention)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'plot_cosine_similarity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_kendall_tau():
    if not KENDALL_FILE.exists():
        return
    with open(KENDALL_FILE, 'r') as f:
        data = json.load(f)
    taus = data['kendall_taus']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(taus, bins=50, color='#0891b2', alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(taus), color=C['secondary'], lw=2, ls='--', label=f'Mean τ = {np.mean(taus):.3f}')
    ax.axvline(0, color='gray', lw=1, alpha=0.5)
    ax.set_xlabel('Kendall τ (Attention vs Gradient Importance)')
    ax.set_ylabel('Count')
    ax.set_title('Transformer: Kendall τ Correlation\n(Attention vs Gradient-Based Importance)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'plot_kendall_tau.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plot_kendall_tau.png")


def plot_summary_dashboard(results, data):
    l1 = [r['l1_difference'] for r in results]
    cos = [r['cosine_similarity'] for r in results]
    same = sum(1 for r in results if r['same_class'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Transformer (DistilBERT): Adversarial Attention — Summary',
                 fontsize=18, fontweight='bold', y=0.98)

    ax = axes[0, 0]
    ax.hist(l1, bins=40, color=C['primary'], alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(l1), color=C['secondary'], lw=2, ls='--', label=f'Mean={np.mean(l1):.3f}')
    ax.set_title('L1 Difference')
    ax.legend()

    ax = axes[0, 1]
    jsd = [r['js_divergence'] for r in results]
    ax.hist(jsd, bins=40, color='#7c3aed', alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(jsd), color=C['secondary'], lw=2, ls='--', label=f'Mean={np.mean(jsd):.3f}')
    ax.set_title('JSD Distribution')
    ax.legend()

    ax = axes[1, 0]
    changed = len(results) - same
    ax.pie([same, changed], labels=[f'Same ({same})', f'Changed ({changed})'],
           colors=[C['positive'], C['negative']], autopct='%1.1f%%', startangle=90)
    ax.set_title('Prediction Stability')

    ax = axes[1, 1]
    ax.axis('off')
    txt = (
        f"Model:          DistilBERT\n"
        f"Samples:        {len(results):,}\n"
        f"Time:           {data.get('total_time_seconds', 0)/60:.1f} min\n\n"
        f"Avg L1 Diff:    {np.mean(l1):.4f}\n"
        f"Avg Cosine Sim: {np.mean(cos):.4f}\n"
        f"Avg JSD:        {np.mean(jsd):.4f}\n\n"
        f"Same Class:     {100*same/len(results):.1f}%\n\n"
        f"CONCLUSION:\n"
        f"Attention is NOT a faithful\n"
        f"explanation in transformers either."
    )
    ax.text(0.1, 0.5, txt, transform=ax.transAxes, fontsize=14,
            va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f4f8', edgecolor='#cbd5e1'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(RESULTS_DIR / 'plot_summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plot_summary_dashboard.png")


def main():
    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found. Run run_full_test_transformer.py first!")
        return
    results, data = load_results()
    print(f"\nGenerating transformer plots...\n")

    plot_l1_histogram(results)
    plot_jsd_histogram(results)
    plot_same_class_pie(results)
    plot_best_method(results)
    plot_cosine_similarity(results)
    plot_kendall_tau()
    plot_summary_dashboard(results, data)


if __name__ == '__main__':
    main()
