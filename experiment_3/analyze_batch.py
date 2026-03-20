import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def extract_metrics(batch_result_file):
    with open(batch_result_file, 'r') as f:
        data = json.load(f)
    
    comp_scores = {1: [], 5: [], 10: []}
    suff_scores = {1: [], 5: [], 10: []}
    
    # Extract scores for each k value
    for review in data['review_details']:
        if 'error' in review:
            continue
        
        # Extract comprehensiveness scores
        if 'comprehensiveness' in review:
            comp_data = review['comprehensiveness']
            if 'results_by_k' in comp_data:
                for k_str, k_results in comp_data['results_by_k'].items():
                    k = int(k_str)
                    if k in comp_scores and 'comprehensiveness' in k_results:
                        comp_scores[k].append(k_results['comprehensiveness'])
        
        # Extract sufficiency scores
        if 'sufficiency' in review:
            suff_data = review['sufficiency']
            if 'results_by_k' in suff_data:
                for k_str, k_results in suff_data['results_by_k'].items():
                    k = int(k_str)
                    if k in suff_scores and 'sufficiency' in k_results:
                        suff_scores[k].append(k_results['sufficiency'])
    
    return comp_scores, suff_scores


def calculate_statistics(scores_dict):
    """Calculate mean, std dev for all k values."""
    stats = {}
    for k, scores in scores_dict.items():
        if scores:
            stats[k] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
    return stats


def plot_individual_histograms(comp_scores, suff_scores, comp_stats, suff_stats, output_dir):
    """Create separate histogram plots for each metric."""
    
    k_values = [1, 5, 10]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot comprehensiveness for each k
    for idx, k in enumerate(k_values):
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = comp_scores[k]
        
        ax.hist(scores, bins=20, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axvline(comp_stats[k]['mean'], color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {comp_stats[k]["mean"]:.4f}')
        
        ax.set_title(f'Comprehensiveness Distribution (k={k})', fontweight='bold', fontsize=14)
        ax.set_xlabel('Comprehensiveness Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics box
        stats_text = (f"Mean: {comp_stats[k]['mean']:.6f}\n"
                     f"Std Dev: {comp_stats[k]['std']:.6f}\n"
                     f"Min: {comp_stats[k]['min']:.6f}\n"
                     f"Max: {comp_stats[k]['max']:.6f}\n"
                     f"Count: {comp_stats[k]['count']}")
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=10, family='monospace')
        
        plt.tight_layout()
        output_file = output_dir / f'comprehensiveness_k{k}_histogram.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Plot sufficiency for each k
    for idx, k in enumerate(k_values):
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = suff_scores[k]
        
        ax.hist(scores, bins=20, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axvline(suff_stats[k]['mean'], color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {suff_stats[k]["mean"]:.4f}')
        
        ax.set_title(f'Sufficiency Distribution (k={k})', fontweight='bold', fontsize=14)
        ax.set_xlabel('Sufficiency Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics box
        stats_text = (f"Mean: {suff_stats[k]['mean']:.6f}\n"
                     f"Std Dev: {suff_stats[k]['std']:.6f}\n"
                     f"Min: {suff_stats[k]['min']:.6f}\n"
                     f"Max: {suff_stats[k]['max']:.6f}\n"
                     f"Count: {suff_stats[k]['count']}")
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                fontsize=10, family='monospace')
        
        plt.tight_layout()
        output_file = output_dir / f'sufficiency_k{k}_histogram.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_comparison(comp_stats, suff_stats, output_dir):
    """Create comparison plots for mean and std dev across k values."""
    
    k_values = [1, 5, 10]
    
    # Comparison of means
    fig, ax = plt.subplots(figsize=(10, 6))
    comp_means = [comp_stats[k]['mean'] for k in k_values]
    suff_means = [suff_stats[k]['mean'] for k in k_values]
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comp_means, width, label='Comprehensiveness', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, suff_means, width, label='Sufficiency', color='#4ECDC4', alpha=0.8)
    
    ax.set_title('Mean Scores Comparison Across k Values', fontweight='bold', fontsize=14)
    ax.set_xlabel('k Value', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_means.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Comparison of standard deviations
    fig, ax = plt.subplots(figsize=(10, 6))
    comp_stds = [comp_stats[k]['std'] for k in k_values]
    suff_stds = [suff_stats[k]['std'] for k in k_values]
    
    bars1 = ax.bar(x - width/2, comp_stds, width, label='Comprehensiveness', color='#45B7D1', alpha=0.8)
    bars2 = ax.bar(x + width/2, suff_stds, width, label='Sufficiency', color='#F7DC6F', alpha=0.8)
    
    ax.set_title('Standard Deviation Comparison Across k Values', fontweight='bold', fontsize=14)
    ax.set_xlabel('k Value', fontsize=12)
    ax.set_ylabel('Std Deviation', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_std_dev.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def print_summary(comp_stats, suff_stats):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("COMPREHENSIVENESS AND SUFFICIENCY ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nCOMPREHENSIVENESS SCORES:")
    print("-" * 70)
    for k in [1, 5, 10]:
        stats = comp_stats[k]
        print(f"\nk={k}:")
        print(f"  Average (Mean):    {stats['mean']:.6f}")
        print(f"  Std Deviation:     {stats['std']:.6f}")
        print(f"  Min:               {stats['min']:.6f}")
        print(f"  Max:               {stats['max']:.6f}")
        print(f"  Count:             {stats['count']}")
    
    print("\n" + "-" * 70)
    print("SUFFICIENCY SCORES:")
    print("-" * 70)
    for k in [1, 5, 10]:
        stats = suff_stats[k]
        print(f"\nk={k}:")
        print(f"  Average (Mean):    {stats['mean']:.6f}")
        print(f"  Std Deviation:     {stats['std']:.6f}")
        print(f"  Min:               {stats['min']:.6f}")
        print(f"  Max:               {stats['max']:.6f}")
        print(f"  Count:             {stats['count']}")
    
    print("\n" + "="*70)


def main():
    batch_result_file = Path(__file__).parent / 'batch_result.json'
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Extract metrics
    comp_scores, suff_scores = extract_metrics(batch_result_file)
    
    # Calculate statistics
    comp_stats = calculate_statistics(comp_scores)
    suff_stats = calculate_statistics(suff_scores)
    
    # Print summary
    print_summary(comp_stats, suff_stats)
    
    # Create and save individual plots
    print("\n📊 Creating individual histogram plots...")
    plot_individual_histograms(comp_scores, suff_scores, comp_stats, suff_stats, output_dir)
    print(f"✓ Saved comprehensiveness histograms (k=1, 5, 10)")
    print(f"✓ Saved sufficiency histograms (k=1, 5, 10)")
    
    # Create and save comparison plots
    print("\n📊 Creating comparison plots...")
    plot_comparison(comp_stats, suff_stats, output_dir)
    print(f"✓ Saved comparison_means.png")
    print(f"✓ Saved comparison_std_dev.png")
    
    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nFiles created:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
