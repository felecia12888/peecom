#!/usr/bin/env python3
"""
PEECOM Framework Schematic Generator - Corrected Version

Creates an accurate, comprehensive schematic of the PEECOM framework addressing:
1. Clear dataset roles (CMOHS vs MotorVD)
2. Proper flow connections and arrows
3. Mathematical feature engineering visualization
4. Algorithm-level flow demonstration
5. Cleaner layout without unnecessary statistics

Engineering perspective: Mechanical + Data Science integration
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon
import numpy as np
import os
from pathlib import Path


def create_corrected_peecom_framework(output_dir="output/figures"):
    """Create corrected and enhanced PEECOM framework schematic"""
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with enhanced layout
    fig, ax = plt.subplots(figsize=(18, 22))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    
    # Define color scheme
    colors = {
        'cmohs': '#2E86AB',           # Blue for CMOHS (primary)
        'motorvd': '#A23B72',         # Pink for MotorVD (validation)
        'preprocessing': '#C73E1D',   # Red-orange for preprocessing
        'physics': '#F18F01',         # Orange for physics features
        'energy': '#E76F51',          # Red-orange for energy
        'efficiency': '#F4A261',      # Yellow-orange for efficiency
        'statistical': '#E9C46A',     # Yellow for statistical
        'composite': '#2A9D8F',       # Teal for composite
        'classifiers': '#264653',     # Dark teal for classifiers
        'selection': '#6A4C93',       # Purple for selection
        'output': '#679436',          # Green for output
        'validation': '#8B5A3C',      # Brown for validation
        'math': '#FF6B6B'             # Coral for mathematical components
    }
    
    # Helper functions
    def create_box(x, y, width, height, color, text, text_color='white', fontsize=10, bold=True):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9
        )
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=text_color)
        return box
    
    def create_arrow(start, end, label='', color='#343A40', style='-|>', linewidth=2):
        arrow = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=20,
                               color=color, linewidth=linewidth)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Title
    ax.text(9, 21, 'PEECOM Framework Architecture', ha='center', va='center',
            fontsize=18, fontweight='bold')
    ax.text(9, 20.5, 'Predictive Energy Efficiency Control and Optimization Model',
            ha='center', va='center', fontsize=12, style='italic')
    
    # ===== DATASET LAYER =====
    ax.text(9, 19.5, 'MULTI-DATASET INTEGRATION', ha='center', va='center',
            fontsize=14, fontweight='bold', color='navy')
    
    # CMOHS Dataset (Primary Training)
    create_box(2, 18.5, 4, 1.2, colors['cmohs'], 
               'CMOHS Dataset\n(Primary Training)\n2,205 hydraulic cycles\n46 sensors → 68 features')
    
    # MotorVD Dataset (Cross-Domain Validation)
    create_box(12, 18.5, 4, 1.2, colors['motorvd'],
               'MotorVD Dataset\n(Cross-Domain Validation)\n107,346 motor samples\nGeneralization testing')
    
    # Dataset Integration
    create_box(7, 18.5, 4, 1.2, colors['validation'],
               'Dataset Integration\n& Validation Strategy\nCross-domain robustness\nGeneralization assessment')
    
    # ===== PREPROCESSING LAYER =====
    create_box(6, 17, 6, 1.2, colors['preprocessing'],
               'Unified Data Preprocessing Pipeline\n• Missing value imputation • Zero correction\n• Outlier removal • Standardization')
    
    # ===== PHASE 1: PHYSICS-INFORMED FEATURE ENGINEERING =====
    # Phase 1 Background with clear boundaries
    phase1_bg = Rectangle((1, 12.5), 16, 4, facecolor='#E3F2FD', alpha=0.3, 
                         edgecolor='navy', linewidth=3, linestyle='-')
    ax.add_patch(phase1_bg)
    ax.text(9, 16.2, 'PHASE 1: Physics-Informed Feature Engineering (46 → 82 features)', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='navy')
    
    # Mathematical Foundation Box
    create_box(1.5, 15.3, 4, 0.8, colors['math'],
               'Mathematical Foundation\nThermodynamic & Hydraulic Principles', fontsize=9)
    
    # Physics Feature Engineering Modules with clear mathematical descriptions
    modules = [
        ('Energy Conservation\nFeatures\n\nP_ij = S_i × S_j\n\nExample:\nP_hydraulic = PS1 × FS1\nP_thermal = TS1 × CE', 
         1.5, 13.5, colors['energy']),
        ('Efficiency Ratio\nFeatures\n\nR_ij = S_i/(S_j + ε)\n\nExample:\nη_pressure = PS1/(PS2+ε)\nη_thermal = TS1/(TS2+ε)', 
         5, 13.5, colors['efficiency']),
        ('Statistical Aggregation\nFeatures\n\nμ, σ, range, CV\n\nExample:\nμ_system = Σ(Si)/n\nσ_variability = √Σ(Si-μ)²/n', 
         8.5, 13.5, colors['statistical']),
        ('System Composite\nFeatures\n\nHolistic Indicators\n\nExample:\nStability = σ_system/μ_system\nRange_util = Smax/Smin', 
         12, 13.5, colors['composite'])
    ]
    
    for text, x, y, color in modules:
        create_box(x, y, 3.5, 1.8, color, text, fontsize=8)
    
    # Enhanced Feature Space with mathematical transformation
    create_box(4, 12.7, 10, 0.8, colors['physics'],
               'Enhanced Feature Space: f(S₁,...,S₄₆) → F(S₁,...,S₄₆,P₁,...,P₃₆) = 82 features\nPhysical interpretation preserved through mathematical transformations')
    
    # ===== PHASE 2: ADAPTIVE MULTI-CLASSIFIER SELECTION =====
    # Phase 2 Background
    phase2_bg = Rectangle((1, 7.5), 16, 4.5, facecolor='#F3E5F5', alpha=0.3, 
                         edgecolor='purple', linewidth=3, linestyle='-')
    ax.add_patch(phase2_bg)
    ax.text(9, 11.7, 'PHASE 2: Adaptive Multi-Classifier Selection (Target-Dependent Optimization)', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='purple')
    
    # Multi-Algorithm Evaluation (clearly connected)
    create_box(1.5, 10.5, 4, 1.5, colors['classifiers'],
               'Multi-Algorithm Evaluation\n(7 Classifiers on CMOHS)\n\n• Random Forest • AdaBoost\n• Neural Networks • Naive Bayes\n• SVM • Logistic Regression\n• Gradient Boosting')
    
    # Physics Benefit Quantification
    create_box(6.5, 10.5, 4, 1.5, colors['selection'],
               'Physics Benefit Quantification\n\nΔ_physics = Acc_enhanced - Acc_raw\nStatistical significance: p < 0.05\nEffect size: Cohen\'s d', fontsize=9)
    
    # Target-Specific Optimal Selection (properly connected)
    create_box(12, 10.5, 4.5, 1.5, colors['selection'],
               'Target-Specific Optimal\nClassifier Selection\n\nOptimal_i = argmax(Δ_physics × Acc_abs)\nData-driven per monitoring task\nAdaptive algorithm choice')
    
    # Algorithm-Level Flow (NEW SECTION)
    create_box(4, 9, 10, 1.2, colors['math'],
               'PEECOM Algorithm-Level Flow\nfor each target t: [Raw → Physics] → [Multi-Eval] → [Best_t] → [Predict_t]\nOptimization: θ* = argmin Σ_t L(y_t, f_θ(X_physics))', fontsize=9)
    
    # ===== MULTI-TARGET OUTPUT LAYER =====
    ax.text(9, 7.8, 'MULTI-TARGET CONDITION MONITORING', ha='center', va='center',
            fontsize=12, fontweight='bold', color='darkgreen')
    
    targets = [
        ('Accumulator\nPressure\n(RandomForest)', 1.5, 6.5),
        ('Cooler\nCondition\n(NaiveBayes)', 4.5, 6.5),
        ('Pump\nLeakage\n(AdaBoost)', 7.5, 6.5),
        ('Valve\nCondition\n(SVM)', 10.5, 6.5),
        ('System\nStability\n(GradBoost)', 13.5, 6.5)
    ]
    
    for target, x, y in targets:
        create_box(x, y, 2.5, 1, colors['output'], target, fontsize=8)
    
    # ===== VALIDATION LAYER =====
    create_box(2, 5, 6, 1.2, colors['validation'],
               'CMOHS Validation Results\n• 10-fold Cross-validation\n• 97-99% accuracy\n• 1.20× ablation resistance', fontsize=9)
    
    create_box(10, 5, 6, 1.2, colors['motorvd'],
               'MotorVD Cross-Domain Validation\n• Generalization testing\n• Framework robustness\n• Domain transferability', fontsize=9)
    
    # ===== CLEAR FLOW ARROWS =====
    
    # Dataset flows
    create_arrow((4, 18.5), (6, 17.5), 'Primary\nTraining', colors['cmohs'])
    create_arrow((14, 18.5), (11, 17.5), 'Cross-Domain\nValidation', colors['motorvd'])
    create_arrow((9, 18.5), (9, 17.5), 'Integration', colors['validation'])
    
    # Preprocessing to Phase 1
    create_arrow((9, 17), (9, 16.2), 'Clean Data')
    
    # Phase 1 internal flows (CORRECTED)
    # Math foundation to all modules
    create_arrow((3.5, 15.3), (3, 14.8), '', colors['math'])
    create_arrow((3.5, 15.5), (6.5, 14.8), '', colors['math'])
    create_arrow((3.5, 15.7), (10, 14.8), '', colors['math'])
    create_arrow((3.5, 15.9), (13.5, 14.8), '', colors['math'])
    
    # All modules to enhanced features
    create_arrow((3, 13.5), (6, 13.3), 'Energy')
    create_arrow((6.7, 13.5), (7.5, 13.3), 'Efficiency')
    create_arrow((10, 13.5), (10.5, 13.3), 'Statistical')
    create_arrow((13.7, 13.5), (12, 13.3), 'Composite')
    
    # Phase 1 to Phase 2
    create_arrow((9, 12.7), (9, 11.7), '82 Enhanced\nFeatures')
    
    # Phase 2 internal flows (CORRECTED)
    create_arrow((5.5, 11.2), (6.5, 11.2), 'Algorithm\nResults')
    create_arrow((10.5, 11.2), (12, 11.2), 'Benefit\nScores')
    
    # Algorithm flow connection
    create_arrow((9, 10.5), (9, 10.2), 'All Algorithms')
    
    # To targets (CORRECTED - from target-specific selection)
    create_arrow((14, 10.5), (9, 7.5), 'Optimal\nModels')
    
    # Validation flows
    create_arrow((5, 6.5), (5, 6.2), 'CMOHS\nResults')
    create_arrow((13, 6.5), (13, 6.2), 'MotorVD\nResults')
    
    # FEEDBACK LOOP (CLARIFIED)
    # This shows robustness validation feeding back to benefit quantification
    create_arrow((5, 5), (8, 10.5), 'Robustness\nFeedback\n(Validation informs\nbenefit assessment)', 
                color='red', style='<-', linewidth=2)
    
    # Add explanation for feedback arrow
    ax.text(6.5, 8, 'Robustness validation results\nfeed back to refine physics\nbenefit quantification', 
            ha='center', va='center', fontsize=8, style='italic', color='red',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='pink', alpha=0.3))
    
    # ===== MATHEMATICAL FEATURE ENGINEERING DETAIL =====
    # Add a mathematical transformation visualization
    math_detail = Rectangle((1, 2.5), 16, 2, facecolor='#FFF3E0', alpha=0.8, 
                           edgecolor='orange', linewidth=2)
    ax.add_patch(math_detail)
    
    ax.text(9, 4.2, 'Mathematical Feature Engineering Innovation', ha='center', va='center',
            fontsize=12, fontweight='bold', color='darkorange')
    
    math_text = '''GEOMETRIC INTERPRETATION: Feature Space Expansion
Raw Space: R⁴⁶ → Enhanced Space: R⁸²
Transform: T(x) = [x, P(x), R(x), S(x), C(x)] where:
• P(x): Power features = {xᵢ × xⱼ | i,j ∈ critical_pairs}
• R(x): Ratio features = {xᵢ/(xⱼ + ε) | i,j ∈ efficiency_pairs} 
• S(x): Statistical features = {μ(x), σ(x), range(x), CV(x)}
• C(x): Composite features = {stability_index, efficiency_ratio, utilization_factor}

PHYSICAL PRESERVATION: Each engineered feature maintains thermodynamic meaning'''
    
    ax.text(9, 3.2, math_text, ha='center', va='center', fontsize=9, color='black')
    
    # Customize plot
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    # Save the corrected diagram
    filename = "peecom_framework_corrected_comprehensive.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
               facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    
    print(f"✅ Corrected PEECOM framework schematic saved: {filepath}")
    return filepath


def create_algorithm_level_flow_diagram(output_dir="output/figures"):
    """Create detailed algorithm-level flow showing PEECOM's innovation"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    
    # Title
    ax.text(8, 9.5, 'PEECOM Algorithm-Level Innovation Flow', ha='center', va='center',
            fontsize=16, fontweight='bold')
    
    # Algorithm steps with mathematical notation
    steps = [
        ('Input: X ∈ R^(n×46)', 1, 8, '#2E86AB'),
        ('Physics Transform:\nT(X) → X_enhanced ∈ R^(n×82)', 4, 8, '#F18F01'),
        ('Multi-Classifier Eval:\n{f₁, f₂, ..., f₇}', 7, 8, '#264653'),
        ('Benefit Quantification:\nΔᵢ = Acc_physics - Acc_raw', 10, 8, '#6A4C93'),
        ('Optimal Selection:\nθ* = argmax(Δᵢ × Accᵢ)', 13, 8, '#6A4C93')
    ]
    
    for text, x, y, color in steps:
        box = FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, 
               fontweight='bold', color='white')
    
    # Flow arrows
    arrows = [((2, 8), (3, 8)), ((5, 8), (6, 8)), ((8, 8), (9, 8)), ((11, 8), (12, 8))]
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=20,
                               color='#343A40', linewidth=2)
        ax.add_patch(arrow)
    
    # Target-specific optimization
    targets = ['Accumulator', 'Cooler', 'Pump', 'Valve', 'Stability']
    for i, target in enumerate(targets):
        x = 2 + i*2.5
        y = 6
        box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6, boxstyle="round,pad=0.05",
                           facecolor='#679436', edgecolor='black', linewidth=1, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, f'{target}\nOptimal: θ*_{i+1}', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
        
        # Arrow from selection to each target
        arrow = FancyArrowPatch((13, 7.5), (x, 6.3), arrowstyle='-|>', mutation_scale=15,
                               color='#343A40', linewidth=1.5)
        ax.add_patch(arrow)
    
    # Mathematical innovation detail
    innovation_text = '''PEECOM's Algorithmic Innovation:
1. Physics-Informed Feature Engineering: T: R⁴⁶ → R⁸² preserving thermodynamic relationships
2. Multi-Classifier Evaluation: Parallel training of 7 algorithms on enhanced features
3. Benefit Quantification: Statistical measurement of physics contribution per algorithm
4. Target-Adaptive Selection: Optimal classifier chosen per monitoring task
5. Robustness Validation: Cross-validation with ablation and statistical testing'''
    
    ax.text(8, 4, innovation_text, ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Performance metrics
    metrics_text = '''Performance Validation:
• Statistical Significance: p < 0.001 (paired t-test)
• Robustness: 1.20× better ablation resistance
• Accuracy: 97-99% across all targets
• Cross-Domain: Validated on MotorVD dataset'''
    
    ax.text(8, 2, metrics_text, ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Save algorithm flow diagram
    filename = "peecom_algorithm_level_flow.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
               facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    
    print(f"✅ PEECOM algorithm-level flow diagram saved: {filepath}")
    return filepath


def main():
    """Generate corrected PEECOM framework diagrams"""
    
    print("🔧 PEECOM Framework Schematic Generator - CORRECTED VERSION")
    print("=" * 65)
    print("Addressing feedback:")
    print("✓ Clear dataset roles (CMOHS primary, MotorVD validation)")
    print("✓ Proper flow connections and arrows")
    print("✓ Mathematical feature engineering visualization") 
    print("✓ Algorithm-level flow demonstration")
    print("✓ Feedback loop clarification")
    print("✓ Removed unnecessary statistics")
    print("=" * 65)
    
    output_dir = "output/figures"
    generated_files = []
    
    try:
        # Generate corrected comprehensive framework
        print("📊 Creating corrected comprehensive framework diagram...")
        comprehensive_file = create_corrected_peecom_framework(output_dir)
        generated_files.append(comprehensive_file)
        
        # Generate algorithm-level flow
        print("🔬 Creating algorithm-level innovation flow...")
        algorithm_file = create_algorithm_level_flow_diagram(output_dir)
        generated_files.append(algorithm_file)
        
        print("\n" + "=" * 65)
        print("✅ CORRECTED PEECOM Framework Visualization Complete!")
        print("=" * 65)
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Generated {len(generated_files)} corrected diagram files:")
        for file in generated_files:
            print(f"   • {Path(file).name}")
        
        print("\n🎯 Corrections Applied:")
        print("• Dataset roles clearly differentiated")
        print("• All flow arrows properly connected")
        print("• Mathematical transformations visualized")
        print("• Algorithm-level innovation demonstrated")
        print("• Feedback loop purpose clarified")
        print("• Engineering perspective integrated")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating corrected diagrams: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())