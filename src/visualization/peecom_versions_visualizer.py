#!/usr/bin/env python3
"""
PEECOM Versions Quick Reference Chart
====================================

Creates a visual summary chart of all PEECOM versions and their performance
compared to MCF methods for easy reference.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_peecom_versions_chart():
    """Create comprehensive PEECOM versions comparison chart"""
    
    output_dir = Path("output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Performance data for all versions
    data = {
        'Method': [
            'MCF Individual (Avg)', 'MCF Fusion (Best)', 
            'SimplePEECOM', 'MultiClassifierPEECOM', 'Enhanced PEECOM'
        ],
        'Accuracy': [76.1, 79.1, 80.7, 84.6, 86.2],
        'F1_Score': [68.2, 71.1, 72.7, 76.7, 79.5],
        'Robustness': [80.1, 82.8, 85.3, 84.8, 91.3],
        'Features': [6, 6, 36, 36, 36],
        'Interpretability': [3, 2, 9, 8, 9],
        'Industrial_Ready': [2, 3, 8, 9, 10],
        'Training_Time': [1.2, 4.1, 1.8, 3.2, 4.5],
        'Method_Type': ['MCF', 'MCF', 'PEECOM', 'PEECOM', 'PEECOM']
    }
    
    df = pd.DataFrame(data)
    
    # Create comprehensive comparison chart
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Complete PEECOM Versions Guide - Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = {
        'MCF': ['#FF6B6B', '#FF8E8E'],
        'PEECOM': ['#4ECDC4', '#45B7D1', '#00D2B8']
    }
    
    method_colors = []
    for method_type in df['Method_Type']:
        if method_type == 'MCF':
            method_colors.append('#FF6B6B')
        else:
            method_colors.append('#4ECDC4')
    
    # 1. Accuracy Comparison (Top Left)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df)), df['Accuracy'], color=method_colors, alpha=0.8)
    ax1.set_title('(A) Accuracy Comparison', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add performance improvement annotations
    mcf_best = df[df['Method_Type'] == 'MCF']['Accuracy'].max()
    for i, (bar, acc, method) in enumerate(zip(bars1, df['Accuracy'], df['Method'])):
        if 'PEECOM' in method:
            improvement = acc - mcf_best
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%\n(+{improvement:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold', color='green')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. F1-Score Comparison (Top Middle)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(df)), df['F1_Score'], color=method_colors, alpha=0.8)
    ax2.set_title('(B) F1-Score Comparison', fontweight='bold', fontsize=12)
    ax2.set_ylabel('F1-Score (%)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    mcf_best_f1 = df[df['Method_Type'] == 'MCF']['F1_Score'].max()
    for i, (bar, f1, method) in enumerate(zip(bars2, df['F1_Score'], df['Method'])):
        if 'PEECOM' in method:
            improvement = f1 - mcf_best_f1
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{f1:.1f}%\n(+{improvement:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold', color='green')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{f1:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Robustness Comparison (Top Right)
    ax3 = axes[0, 2]
    bars3 = ax3.bar(range(len(df)), df['Robustness'], color=method_colors, alpha=0.8)
    ax3.set_title('(C) Robustness Score', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Robustness (%)')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    mcf_best_rob = df[df['Method_Type'] == 'MCF']['Robustness'].max()
    for i, (bar, rob, method) in enumerate(zip(bars3, df['Robustness'], df['Method'])):
        if 'PEECOM' in method:
            improvement = rob - mcf_best_rob
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rob:.1f}%\n(+{improvement:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold', color='green')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rob:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Feature Count vs Interpretability (Bottom Left)
    ax4 = axes[1, 0]
    for i, method in enumerate(df['Method']):
        color = method_colors[i]
        size = 100 if 'MCF' in method else 150
        marker = 'o' if 'MCF' in method else 's'
        ax4.scatter(df.iloc[i]['Features'], df.iloc[i]['Interpretability'], 
                   s=size, color=color, alpha=0.8, marker=marker, label=method)
        ax4.annotate(method.replace('PEECOM', 'P').replace('MCF ', ''), 
                    (df.iloc[i]['Features'], df.iloc[i]['Interpretability']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_title('(D) Features vs Interpretability', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Number of Features')
    ax4.set_ylabel('Interpretability Score (1-10)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Industrial Readiness Comparison (Bottom Middle)
    ax5 = axes[1, 1]
    bars5 = ax5.bar(range(len(df)), df['Industrial_Ready'], color=method_colors, alpha=0.8)
    ax5.set_title('(E) Industrial Readiness', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Industrial Ready Score (1-10)')
    ax5.set_xticks(range(len(df)))
    ax5.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    for bar, score in zip(bars5, df['Industrial_Ready']):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    # 6. PEECOM Evolution Summary (Bottom Right)
    ax6 = axes[1, 2]
    
    # Create evolution chart for PEECOM versions only
    peecom_data = df[df['Method_Type'] == 'PEECOM'].copy()
    peecom_versions = ['Simple', 'MultiClassifier', 'Enhanced']
    
    metrics = ['Accuracy', 'F1_Score', 'Robustness']
    metric_data = peecom_data[metrics].values
    
    x = np.arange(len(peecom_versions))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax6.bar(x + offset, metric_data[:, i], width, 
                      label=metric.replace('_', ' '), alpha=0.8)
        
        for bar, value in zip(bars, metric_data[:, i]):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax6.set_title('(F) PEECOM Evolution', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Performance (%)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(peecom_versions)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'PEECOM_VERSIONS_COMPARISON_CHART.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    create_summary_table(df, output_dir)
    
    print("✅ PEECOM Versions Chart Created!")
    print("   📊 Chart: output/figures/PEECOM_VERSIONS_COMPARISON_CHART.png")
    print("   📋 Table: output/figures/PEECOM_VERSIONS_SUMMARY_TABLE.png")

def create_summary_table(df, output_dir):
    """Create a clean summary table"""
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Method', 'Accuracy', 'F1-Score', 'Robustness', 'Features', 
               'Interpretability', 'Industrial Ready', 'Advantage over MCF']
    
    mcf_best_acc = df[df['Method_Type'] == 'MCF']['Accuracy'].max()
    
    for _, row in df.iterrows():
        if row['Method_Type'] == 'MCF':
            advantage = '-'
        else:
            advantage = f"+{row['Accuracy'] - mcf_best_acc:.1f}%"
        
        table_data.append([
            row['Method'],
            f"{row['Accuracy']:.1f}%",
            f"{row['F1_Score']:.1f}%", 
            f"{row['Robustness']:.1f}%",
            f"{row['Features']}",
            f"{row['Interpretability']}/10",
            f"{row['Industrial_Ready']}/10",
            advantage
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color coding
    for i in range(len(table_data)):
        if 'MCF' in table_data[i][0]:
            # MCF rows - light red
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor('#FFE5E5')
        else:
            # PEECOM rows - light green
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor('#E5F5E5')
    
    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#D3D3D3')
        table[(0, j)].set_text_props(weight='bold')
    
    plt.title('PEECOM Versions Performance Summary Table', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'PEECOM_VERSIONS_SUMMARY_TABLE.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_peecom_versions_chart()