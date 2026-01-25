"""
Visualize HyDE vs Baseline comparison metrics
Creates bar charts for easy understanding
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime


def load_comparison_results(json_file):
    """Load comparison results from JSON"""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_comparison_charts(results, save_path="eval_results"):
    """
    Create visualization comparing Baseline vs HyDE
    
    Args:
        results: Comparison results dict
        save_path: Directory to save images
    """
    
    agg = results["aggregate"]
    
    # Extract metrics (convert to percentages)
    metrics_names = ['Faithfulness', 'F1 Score', 'answer_correctness']
    baseline_scores = [
        agg['baseline']['avg_faithfulness'] * 100,
        agg['baseline']['avg_f1_score'] * 100,
        agg['baseline']['avg_answer_correctness'] * 100
    ]
    hyde_scores = [
        agg['hyde']['avg_faithfulness'] * 100,
        agg['hyde']['avg_f1_score'] * 100,
        agg['hyde']['avg_answer_correctness'] * 100
    ]
    
    # Latency (in seconds)
    latency_baseline = agg['baseline']['avg_latency']
    latency_hyde = agg['hyde']['avg_latency']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('HyDE vs Baseline RAG - Performance Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ============================================
    # Chart 1: Accuracy Metrics (%)
    # ============================================
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_scores, width, 
                    label='Baseline', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, hyde_scores, width, 
                    label='HyDE', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Customize chart 1
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Metrics', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=11)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='Target (90%)')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement arrows
    for i, (b, h) in enumerate(zip(baseline_scores, hyde_scores)):
        diff = h - b
        color = 'green' if diff > 0 else 'red'
        symbol = '↑' if diff > 0 else '↓'
        ax1.text(i, max(b, h) + 3, f'{symbol} {abs(diff):.1f}%', 
                ha='center', fontsize=9, color=color, fontweight='bold')
    
    # ============================================
    # Chart 2: Latency (seconds)
    # ============================================
    
    latency_names = ['Baseline', 'HyDE']
    latency_values = [latency_baseline, latency_hyde]
    colors = ['#3498db', '#e74c3c']
    
    bars3 = ax2.bar(latency_names, latency_values, 
                    color=colors, alpha=0.8, edgecolor='black', width=0.6)
    
    # Customize chart 2
    ax2.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Latency Comparison', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, max(latency_values) * 1.3)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=3, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Target (3s)')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add latency overhead annotation
    overhead = latency_hyde - latency_baseline
    ax2.annotate(f'+{overhead:.2f}s overhead', 
                xy=(1, latency_hyde), xytext=(1.3, latency_hyde - 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    # Overall layout
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{timestamp}_hyde_comparison.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved to: {save_file}")
    
    # Display
    plt.show()
    
    return save_file


def create_summary_table(results):
    """
    Create a summary table visualization
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    
    agg = results["aggregate"]
    
    # Prepare table data
    table_data = [
        ['Metric', 'Baseline', 'HyDE', 'Improvement', 'Winner'],
        ['Faithfulness', 
         f"{agg['baseline']['avg_faithfulness']*100:.1f}%",
         f"{agg['hyde']['avg_faithfulness']*100:.1f}%",
         f"{agg['improvement']['faithfulness']:+.1f}%",
         '🏆 HyDE' if agg['improvement']['faithfulness'] > 0 else '🏆 Baseline'],
        ['F1 Score',
         f"{agg['baseline']['avg_f1_score']*100:.1f}%",
         f"{agg['hyde']['avg_f1_score']*100:.1f}%",
         f"{agg['improvement']['f1_score']:+.1f}%",
         '🏆 HyDE' if agg['improvement']['f1_score'] > 0 else '🏆 Baseline'],
        ['Answer Correctness',
         f"{agg['baseline']['avg_answer_correctness']*100:.1f}%",
         f"{agg['hyde']['avg_answer_correctness']*100:.1f}%",
         f"{agg['improvement']['answer_correctness']:+.1f}%",
         '🏆 HyDE' if agg['improvement']['answer_correctness'] > 0 else '🏆 Baseline'],
        ['Latency',
         f"{agg['baseline']['avg_latency']:.2f}s",
         f"{agg['hyde']['avg_latency']:.2f}s",
         f"+{agg['improvement']['latency_overhead']:.2f}s",
         '🏆 Baseline']
    ]
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white')
    
    # Color improvement column
    for i in range(1, 5):
        cell = table[(i, 3)]
        improvement = float(table_data[i][3].replace('%', '').replace('+', '').replace('s', ''))
        if improvement > 0:
            cell.set_facecolor('#d4edda')  # Light green
        else:
            cell.set_facecolor('#f8d7da')  # Light red
    
    plt.title('HyDE vs Baseline - Summary Table', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = os.path.join("eval_results", f"{timestamp}_hyde_table.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ Table saved to: {save_file}")
    
    plt.show()


if __name__ == "__main__":
    # Load latest comparison results
    import glob
    
    json_files = glob.glob("eval_results/*_hyde_comparison.json")
    if not json_files:
        print("❌ No comparison results found. Run compare_hyde.py first.")
        exit(1)
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📂 Loading: {latest_file}\n")
    
    results = load_comparison_results(latest_file)
    
    # Create visualizations
    print("📊 Creating comparison charts...")
    create_comparison_charts(results)
    
    print("\n📋 Creating summary table...")
    create_summary_table(results)
    
    print("\n✅ Visualization complete!")