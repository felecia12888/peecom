#!/usr/bin/env python3
"""
PEECOM Simplified Framework Diagram Generator

Creates a clean, publication-ready schematic diagram of the PEECOM framework
using Graphviz for vector graphics output suitable for manuscripts.

Usage:
    python peecom_simple_schematic.py
    python peecom_simple_schematic.py --output output/figures --format svg
"""

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Error: Graphviz not installed. Install with: pip install graphviz")
    exit(1)

import os
import argparse
from pathlib import Path


def create_peecom_schematic(output_dir="output/figures", format="png"):
    """Create a comprehensive PEECOM framework schematic using Graphviz"""
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the diagram with professional styling
    diagram = Digraph("PEECOM_Framework", format=format)
    diagram.attr(rankdir="TB", splines="ortho", nodesep="0.6", ranksep="0.8")
    diagram.attr('node', style='filled', fontname='Arial', fontsize='10')
    diagram.attr('edge', fontname='Arial', fontsize='9')
    
    # Define color scheme (professional hydraulic engineering theme)
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
    
    # Input Layer
    diagram.node("input", 
                "Raw Hydraulic Sensor Data\\n(46 sensors)\\l\\l"
                "• Pressure sensors (PS1-PS6)\\l"
                "• Temperature sensors (TS1-TS4)\\l" 
                "• Flow sensors (FS1-FS2)\\l"
                "• Power sensor (EPS1)\\l"
                "• Efficiency indicators (CE,CP,SE)\\l",
                shape="box", fillcolor=colors['input'], fontcolor="white")
    
    # Preprocessing
    diagram.node("preprocess",
                "Data Preprocessing Pipeline\\l\\l"
                "• Missing value imputation\\l"
                "• Zero value correction (sensor failures)\\l"
                "• Outlier detection and removal\\l" 
                "• Sensor drift correction\\l"
                "• Feature standardization\\l",
                shape="box", fillcolor=colors['preprocessing'], fontcolor="white")
    
    # Phase 1 Cluster: Physics-Informed Feature Engineering
    with diagram.subgraph(name='cluster_phase1') as phase1:
        phase1.attr(label='PHASE 1: Physics-Informed Feature Engineering\\n(46 → 82 features)',
                   style='filled,rounded', fillcolor='#F0F8FF', fontsize='12', fontcolor='navy')
        
        # Physics feature engineering modules
        phase1.node("energy_features",
                   "Energy Conservation Features\\l\\l"
                   "P_ij = S_i × S_j\\l\\l"
                   "• Hydraulic power (pressure × flow)\\l"
                   "• Thermal power (temperature × efficiency)\\l"
                   "• Electro-hydraulic coupling\\l",
                   shape="ellipse", fillcolor=colors['energy'], fontcolor="white")
        
        phase1.node("efficiency_features", 
                   "Efficiency Ratio Features\\l\\l"
                   "R_ij = S_i / (S_j + ε)\\l\\l"
                   "• Pressure differentials (ΔP)\\l"
                   "• Thermal gradients (ΔT)\\l" 
                   "• Volumetric efficiency ratios\\l",
                   shape="ellipse", fillcolor=colors['efficiency'], fontcolor="white")
        
        phase1.node("statistical_features",
                   "Statistical Aggregation Features\\l\\l"
                   "μ, σ, range, CV\\l\\l"
                   "• Mean response (system state)\\l"
                   "• Standard deviation (variability)\\l"
                   "• Range indicators (extremes)\\l",
                   shape="ellipse", fillcolor=colors['statistical'], fontcolor="white")
        
        phase1.node("composite_features",
                   "System-Level Composite Features\\l\\l"
                   "Holistic indicators\\l\\l"
                   "• Operational range utilization\\l"
                   "• Coefficient of variation\\l"
                   "• Energy efficiency index\\l",
                   shape="ellipse", fillcolor=colors['composite'], fontcolor="white")
        
        # Enhanced feature space
        phase1.node("enhanced_features",
                   "Enhanced Feature Space\\n(82 physics-informed features)\\n\\n"
                   "Expanded from 46 → 82 dimensions\\n"
                   "Physical interpretation preserved",
                   shape="box", fillcolor=colors['physics'], fontcolor="white")
    
    # Phase 2 Cluster: Adaptive Multi-Classifier Selection  
    with diagram.subgraph(name='cluster_phase2') as phase2:
        phase2.attr(label='PHASE 2: Adaptive Multi-Classifier Selection\\n(Target-dependent optimization)',
                   style='filled,rounded', fillcolor='#F5F0FF', fontsize='12', fontcolor='purple')
        
        # Multi-algorithm evaluation
        phase2.node("multi_algorithms",
                   "Multi-Algorithm Evaluation\\l\\l"
                   "7 classifier families:\\l"
                   "• Random Forest (ensemble)\\l"
                   "• AdaBoost (boosting)\\l"
                   "• Neural Networks (deep learning)\\l"
                   "• Naive Bayes (probabilistic)\\l"
                   "• SVM (margin optimization)\\l"
                   "• Logistic Regression (linear)\\l"
                   "• Gradient Boosting (ensemble)\\l",
                   shape="diamond", fillcolor=colors['classifiers'], fontcolor="white")
        
        # Physics benefit quantification
        phase2.node("benefit_quantification",
                   "Physics Benefit Quantification\\l\\l"
                   "ΔPerformance = Acc_physics - Acc_raw\\l\\l"
                   "• Statistical significance testing\\l"
                   "• Paired t-tests (p < 0.05)\\l"
                   "• Effect size measurement\\l",
                   shape="hexagon", fillcolor=colors['selection'], fontcolor="white")
        
        # Automatic selection
        phase2.node("auto_selection",
                   "Automatic Classifier Selection\\l\\l"
                   "Optimal = max(Physics_Benefit × Performance)\\l\\l"
                   "• Target-dependent optimization\\l"
                   "• Data-driven selection\\l"
                   "• Performance-benefit trade-off\\l",
                   shape="diamond", fillcolor=colors['selection'], fontcolor="white")
    
    # Multi-target output
    diagram.node("multi_target_output",
                "Multi-Target Condition Monitoring\\l\\l"
                "Simultaneous assessment of:\\l"
                "• Accumulator pressure condition\\l"
                "• Cooler efficiency status\\l"
                "• Pump leakage detection\\l"
                "• Valve condition monitoring\\l"
                "• System stability assessment\\l",
                shape="box", fillcolor=colors['output'], fontcolor="white")
    
    # Validation and evaluation
    diagram.node("validation",
                "Comprehensive Validation Framework\\l\\l"
                "• 10-fold stratified cross-validation\\l"
                "• Statistical significance testing\\l"
                "• Ablation resistance analysis (1.20× improvement)\\l"
                "• Permutation importance validation\\l"
                "• Robustness under sensor failures\\l"
                "• Performance: 97-99% accuracy\\l",
                shape="box", fillcolor=colors['validation'], fontcolor="white")
    
    # Add all edges to show data flow
    diagram.edge("input", "preprocess", label="Raw sensor\\nreadings")
    
    # Preprocessing to all physics feature modules
    diagram.edge("preprocess", "energy_features", label="Clean data")
    diagram.edge("preprocess", "efficiency_features")
    diagram.edge("preprocess", "statistical_features") 
    diagram.edge("preprocess", "composite_features")
    
    # All physics features to enhanced feature space
    diagram.edge("energy_features", "enhanced_features", label="Power\\nrelationships")
    diagram.edge("efficiency_features", "enhanced_features", label="Efficiency\\nratios")
    diagram.edge("statistical_features", "enhanced_features", label="Statistical\\nmeasures")
    diagram.edge("composite_features", "enhanced_features", label="System-level\\nindicators")
    
    # Enhanced features to Phase 2
    diagram.edge("enhanced_features", "multi_algorithms", label="82 physics-enhanced\\nfeatures")
    
    # Phase 2 internal flow
    diagram.edge("multi_algorithms", "benefit_quantification", label="Algorithm\\nperformance")
    diagram.edge("benefit_quantification", "auto_selection", label="Physics benefit\\nquantification")
    
    # Final outputs
    diagram.edge("auto_selection", "multi_target_output", label="Optimal classifier\\nper target")
    diagram.edge("multi_target_output", "validation", label="Prediction\\nresults")
    
    # Feedback loop for robustness
    diagram.edge("validation", "benefit_quantification", 
                style="dashed", color="red", label="Robustness\\nfeedback")
    
    # Add framework statistics as a note
    diagram.node("stats",
                "PEECOM Framework Statistics\\l\\l"
                "• Dataset: 2,205 hydraulic cycles\\l"
                "• Feature expansion: 46 → 82 (+78%)\\l" 
                "• Classifiers evaluated: 7 algorithms\\l"
                "• Targets monitored: 5 conditions\\l"
                "• Accuracy achieved: 97-99%\\l"
                "• Robustness improvement: 1.20×\\l"
                "• Statistical significance: p < 0.001\\l",
                shape="note", fillcolor="lightyellow", fontcolor="black")
    
    # Render the diagram
    output_file = output_path / "peecom_framework_schematic"
    diagram.render(str(output_file), cleanup=True)
    
    print(f"✅ PEECOM framework schematic generated: {output_file}.{format}")
    return f"{output_file}.{format}"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate PEECOM framework schematic')
    parser.add_argument('--output', default='output/figures', 
                       help='Output directory (default: output/figures)')
    parser.add_argument('--format', choices=['png', 'svg', 'pdf'], default='png',
                       help='Output format (default: png)')
    
    args = parser.parse_args()
    
    if not HAS_GRAPHVIZ:
        print("❌ Error: Graphviz not installed")
        print("💡 Install with: pip install graphviz")
        return 1
    
    print("🎯 Generating PEECOM Framework Schematic...")
    print("=" * 50)
    
    # Generate the schematic
    output_file = create_peecom_schematic(args.output, args.format)
    
    print("=" * 50)
    print("🎉 PEECOM schematic generation complete!")
    print(f"📄 File saved: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())