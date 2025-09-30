#!/usr/bin/env python3
"""
PEECOM Framework Schematic Generator - Robust Version

Creates a comprehensive schematic of the PEECOM framework using matplotlib
with guaranteed compatibility and no external dependencies beyond basic packages.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np
import os
from pathlib import Path


def create_peecom_framework_schematic(output_dir="output/figures"):
    """Create a comprehensive PEECOM framework schematic diagram"""
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with professional layout
    fig, ax = plt.subplots(figsize=(16, 20))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    
    # Define color scheme
    colors = {
        'input': '#2E86AB',           # Blue for input
        'preprocessing': '#C73E1D',   # Red-orange for preprocessing
        'physics': '#F18F01',         # Orange for physics features
        'energy': '#E76F51',          # Red-orange for energy
        'efficiency': '#F4A261',      # Yellow-orange for efficiency
        'statistical': '#E9C46A',     # Yellow for statistical
        'composite': '#2A9D8F',       # Teal for composite
        'classifiers': '#264653',     # Dark teal for classifiers
        'selection': '#6A4C93',       # Purple for selection
        'output': '#679436',          # Green for output
        'validation': '#8B5A3C'       # Brown for validation
    }
    
    # Helper function to create rounded boxes
    def create_box(x, y, width, height, color, text, text_color='white', fontsize=10):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)
        return box
    
    # Helper function to create arrows
    def create_arrow(start, end, label='', color='#343A40'):
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=20,
                               color=color, linewidth=2)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2
            ax.text(mid_x + 0.2, mid_y, label, fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Title
    ax.text(8, 19, 'PEECOM Framework Architecture', ha='center', va='center',
            fontsize=18, fontweight='bold')
    ax.text(8, 18.5, 'Predictive Energy Efficiency Control and Optimization Model',
            ha='center', va='center', fontsize=12, style='italic')
    
    # Input Layer
    create_box(6, 17, 4, 1.2, colors['input'], 
               'Raw Hydraulic Sensor Data\n(46 sensors)\n\nPS1-PS6, TS1-TS4, FS1-FS2\nEPS1, CE, CP, SE')
    
    # Preprocessing
    create_box(6, 15, 4, 1.5, colors['preprocessing'],
               'Data Preprocessing Pipeline\n\n• Missing value imputation\n• Zero value correction\n• Outlier removal\n• Standardization')
    
    # Phase 1 Background
    phase1_bg = Rectangle((1, 10), 14, 4, facecolor='#E3F2FD', alpha=0.3, edgecolor='navy', linewidth=2)
    ax.add_patch(phase1_bg)
    ax.text(8, 13.7, 'PHASE 1: Physics-Informed Feature Engineering', ha='center', va='center',
            fontsize=14, fontweight='bold', color='navy')
    
    # Physics Feature Engineering Modules
    modules = [
        ('Energy Conservation\nFeatures\n\nP_ij = S_i × S_j\n(Power relationships)', 1.5, 11.5, colors['energy']),
        ('Efficiency Ratio\nFeatures\n\nR_ij = S_i/(S_j + ε)\n(Performance ratios)', 4.5, 11.5, colors['efficiency']),
        ('Statistical Aggregation\nFeatures\n\nμ, σ, range, CV\n(System behavior)', 7.5, 11.5, colors['statistical']),
        ('System-Level\nComposite Features\n\nStability metrics\nEfficiency index', 10.5, 11.5, colors['composite'])
    ]
    
    for text, x, y, color in modules:
        create_box(x, y, 3, 1.8, color, text, fontsize=9)
    
    # Enhanced Feature Space
    create_box(4, 10.2, 8, 1, colors['physics'],
               'Enhanced Feature Space (82 physics-informed features)\nExpanded from 46 → 82 dimensions with physical interpretation')
    
    # Phase 2 Background
    phase2_bg = Rectangle((1, 5.5), 14, 4, facecolor='#F3E5F5', alpha=0.3, edgecolor='purple', linewidth=2)
    ax.add_patch(phase2_bg)
    ax.text(8, 9.2, 'PHASE 2: Adaptive Multi-Classifier Selection', ha='center', va='center',
            fontsize=14, fontweight='bold', color='purple')
    
    # Multi-Algorithm Evaluation
    create_box(1.5, 7.5, 4, 1.5, colors['classifiers'],
               'Multi-Algorithm Evaluation\n\n• Random Forest\n• AdaBoost\n• Neural Networks\n• Naive Bayes\n• SVM\n• Logistic Regression\n• Gradient Boosting')
    
    # Physics Benefit Quantification
    create_box(6.5, 7.5, 3.5, 1.5, colors['selection'],
               'Physics Benefit\nQuantification\n\nΔ = Acc_physics - Acc_raw\nStatistical significance\ntesting (p < 0.05)')
    
    # Automatic Selection
    create_box(11, 7.5, 3.5, 1.5, colors['selection'],
               'Automatic Classifier\nSelection\n\nOptimal = max(Benefit × Performance)\nTarget-dependent\noptimization')
    
    # Enhanced Feature Space
    create_box(4, 6, 8, 1, colors['physics'],
               'Target-Specific Optimal Classifier Selection\nData-driven algorithm choice per monitoring task')
    
    # Multi-Target Output
    targets = ['Accumulator\nPressure', 'Cooler\nCondition', 'Pump\nLeakage', 'Valve\nCondition', 'System\nStability']
    for i, target in enumerate(targets):
        create_box(1.5 + i*2.8, 4, 2.5, 1, colors['output'], target, fontsize=9)
    
    # Validation Framework
    create_box(3, 2, 10, 1.5, colors['validation'],
               'Comprehensive Validation Framework\n\n• 10-fold Cross-validation • Statistical significance testing\n• Ablation resistance (1.20× improvement) • Permutation importance\n• Performance: 97-99% accuracy • Robustness under sensor failures')
    
    # Add arrows showing data flow
    # Input to preprocessing
    create_arrow((8, 17), (8, 16.5), 'Raw data')
    
    # Preprocessing to physics modules
    create_arrow((8, 15), (8, 14.2), 'Clean data')
    
    # Physics modules to enhanced features
    for i in range(4):
        start_x = 3 + i*3
        create_arrow((start_x, 11.5), (8, 11.2))
    
    # Enhanced features to Phase 2
    create_arrow((8, 10.2), (8, 9.5), '82 features')
    
    # Phase 2 internal flow
    create_arrow((3.5, 8.2), (6.5, 8.2))
    create_arrow((10, 8.2), (11, 8.2))
    
    # To target outputs
    create_arrow((8, 6), (8, 5), 'Optimal models')
    
    # To validation
    create_arrow((8, 4), (8, 3.5), 'Results')
    
    # Feedback loop
    create_arrow((3, 2.7), (6.5, 7.5), 'Robustness\nfeedback', color='red')
    
    # Framework statistics box
    stats_text = '''PEECOM Framework Performance Statistics
    
• Primary Dataset: 2,205 hydraulic system cycles (CMOHS) for main analysis
• Cross-Domain Validation: 107,346 motor samples (MotorVD) for generalization testing
• Feature Engineering: 46 → 82 physics-informed features (+78% expansion)
• Algorithm Evaluation: 7 different machine learning classifiers tested on CMOHS
• Multi-Target Monitoring: 5 simultaneous condition assessments (hydraulic systems)
• Accuracy Achievement: 97-99% across all targets on primary hydraulic dataset
• Robustness Improvement: 1.20× better ablation resistance vs. conventional ML
• Statistical Validation: p < 0.001 significance, 10-fold cross-validation
• Cross-Domain Performance: Successful adaptation to motor vibration data (MotorVD)
• Industrial Deployment: Real-time capable with <100ms prediction time'''
    
    create_box(0.5, 0.2, 15, 1.5, 'lightblue', stats_text, 'black', fontsize=9)
    
    # Customize plot
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Save the diagram
    filename = "peecom_framework_comprehensive_schematic.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
               facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    
    print(f"✅ PEECOM framework schematic saved: {filepath}")
    return filepath


def create_simplified_flow_diagram(output_dir="output/figures"):
    """Create a simplified flow diagram showing the main PEECOM process"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    
    # Define positions for flow diagram
    positions = [
        (1, 4, 'Raw Sensor\nData\n(46 sensors)', '#2E86AB'),
        (3.5, 4, 'Data\nPreprocessing', '#C73E1D'),
        (6, 4, 'Physics-Enhanced\nFeature Engineering\n(82 features)', '#F18F01'),
        (9, 4, 'Multi-Classifier\nEvaluation\n(7 algorithms)', '#264653'),
        (11.5, 4, 'Optimal Model\nSelection', '#6A4C93'),
        (6, 2, 'Multi-Target\nCondition Monitoring', '#679436'),
        (9, 2, 'Performance\nValidation\n(97-99% accuracy)', '#8B5A3C')
    ]
    
    # Create boxes and labels
    for x, y, text, color in positions:
        box = FancyBboxPatch(
            (x-0.7, y-0.5), 1.4, 1,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    
    # Add flow arrows
    flow_arrows = [
        ((1.7, 4), (2.8, 4)),      # Raw data to preprocessing
        ((4.2, 4), (5.3, 4)),      # Preprocessing to physics features  
        ((6.7, 4), (8.3, 4)),      # Physics features to classifiers
        ((9.7, 4), (10.8, 4)),     # Classifiers to selection
        ((11.5, 3.5), (9.7, 2.5)), # Selection to monitoring
        ((8.3, 2), (9.7, 2)),      # Monitoring to validation
        ((9, 1.5), (6, 3.5))       # Validation feedback
    ]
    
    for start, end in flow_arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=20,
                               color='#343A40', linewidth=2)
        ax.add_patch(arrow)
    
    # Title
    ax.text(6.5, 6, 'PEECOM Framework Flow Diagram', ha='center', va='center',
            fontsize=16, fontweight='bold')
    
    # Key benefits annotation
    benefits_text = '''Key PEECOM Benefits:
• Physics-informed feature engineering preserves domain knowledge
• Adaptive classifier selection optimizes performance per target
• 1.20× better robustness to sensor failures vs. conventional ML
• 97-99% accuracy across 5 simultaneous monitoring targets'''
    
    ax.text(2, 0.5, benefits_text, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Save the diagram
    filename = "peecom_simplified_flow_diagram.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
               facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    
    print(f"✅ PEECOM simplified flow diagram saved: {filepath}")
    return filepath


def main():
    """Generate all PEECOM framework diagrams"""
    
    print("🎯 PEECOM Framework Schematic Generator")
    print("=" * 50)
    
    # Create output directory
    output_dir = "output/figures"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    try:
        # Generate comprehensive schematic
        print("📊 Creating comprehensive framework schematic...")
        comprehensive_file = create_peecom_framework_schematic(output_dir)
        generated_files.append(comprehensive_file)
        
        # Generate simplified flow diagram
        print("📈 Creating simplified flow diagram...")
        flow_file = create_simplified_flow_diagram(output_dir)
        generated_files.append(flow_file)
        
        print("\n" + "=" * 50)
        print("✅ PEECOM Framework Visualization Complete!")
        print("=" * 50)
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Generated {len(generated_files)} diagram files:")
        for file in generated_files:
            print(f"   • {Path(file).name}")
        
        # Add these to manuscript
        print("\n💡 Usage for Manuscript:")
        print("• Use comprehensive schematic for framework architecture section")
        print("• Use simplified flow diagram for methodology overview")
        print("• Both diagrams are publication-ready at 300 DPI")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating diagrams: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())