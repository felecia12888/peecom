#!/usr/bin/env python3
"""
PEECOM Framework Schematic Visualizer

Generates comprehensive schematic diagrams for the PEECOM (Predictive Energy Efficiency 
Control and Optimization Model) framework architecture, showing the two-phase system:
Phase 1: Physics-Informed Feature Engineering
Phase 2: Adaptive Multi-Classifier Selection

Usage:
    python peecom_framework_visualizer.py --output output/figures --style detailed
    python peecom_framework_visualizer.py --style simplified --format svg
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, ConnectionPatch
from matplotlib.patches import Ellipse, Polygon, FancyArrowPatch
import matplotlib.colors as mcolors
import numpy as np
import os
import argparse
import seaborn as sns
from pathlib import Path

# Try to import graphviz for alternative diagram generation
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Warning: Graphviz not installed. Some diagram features will be limited.")


class PEECOMFrameworkVisualizer:
    """Comprehensive PEECOM Framework Visualization System"""
    
    def __init__(self, output_dir="output/figures"):
        """Initialize the visualizer with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define PEECOM color scheme (professional hydraulic/industrial theme)
        self.colors = {
            'input': '#2E86AB',           # Professional blue for input data
            'sensors': '#A23B72',         # Deep pink for sensors
            'preprocessing': '#C73E1D',   # Red-orange for preprocessing
            'physics_features': '#F18F01', # Orange for physics features
            'energy': '#E76F51',          # Red-orange for energy features
            'efficiency': '#F4A261',      # Yellow-orange for efficiency
            'statistical': '#E9C46A',     # Yellow for statistical features
            'composite': '#2A9D8F',       # Teal for composite features
            'classifiers': '#264653',     # Dark teal for classifiers
            'selection': '#6A4C93',       # Purple for selection
            'output': '#679436',          # Green for output
            'evaluation': '#5D737E',      # Blue-gray for evaluation
            'background': '#F8F9FA',      # Light background
            'connections': '#343A40'      # Dark connections
        }
    
    def create_comprehensive_framework_diagram(self, style='detailed'):
        """Create a comprehensive PEECOM framework diagram"""
        
        # Setup figure with appropriate size
        if style == 'detailed':
            fig, ax = plt.subplots(figsize=(20, 14))
        else:
            fig, ax = plt.subplots(figsize=(16, 10))
        
        # Set background
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        # Define layout parameters
        margin = 1
        total_width = 18
        total_height = 12
        
        # Phase 1: Physics-Informed Feature Engineering (Left Side)
        self._draw_phase1_physics_engineering(ax, margin, total_height, style)
        
        # Phase 2: Adaptive Multi-Classifier Selection (Right Side)
        self._draw_phase2_classifier_selection(ax, total_width/2 + 1, total_height, style)
        
        # Draw data flow connections
        self._draw_data_flow_connections(ax, total_width, total_height)
        
        # Add title and metadata
        self._add_title_and_metadata(ax, total_width, total_height, style)
        
        # Customize plot
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save the diagram
        filename = f"peecom_framework_comprehensive_{style}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        print(f"✅ Comprehensive PEECOM framework diagram saved: {filepath}")
        return filepath
    
    def _draw_phase1_physics_engineering(self, ax, start_x, total_height, style):
        """Draw Phase 1: Physics-Informed Feature Engineering"""
        
        # Phase 1 Title Box
        phase1_box = FancyBboxPatch(
            (start_x, total_height - 1.5), 8, 1,
            boxstyle="round,pad=0.1", 
            facecolor=self.colors['physics_features'],
            edgecolor='black', linewidth=2
        )
        ax.add_patch(phase1_box)
        ax.text(start_x + 4, total_height - 1, 'PHASE 1: Physics-Informed Feature Engineering',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Input Data Section
        input_box = self._create_rounded_box(start_x, total_height - 3, 3.5, 1,
                                           self.colors['input'], 'Raw Sensor Data\n(46 sensors)')
        ax.add_patch(input_box)
        ax.text(start_x + 1.75, total_height - 2.5, 'Raw Sensor Data\n(46 sensors)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Sensor Categories (if detailed style)
        if style == 'detailed':
            sensors = [
                'Pressure (PS1-PS6)', 'Temperature (TS1-TS4)', 
                'Flow (FS1-FS2)', 'Power (EPS1)', 'Efficiency (CE,CP,SE)'
            ]
            for i, sensor in enumerate(sensors):
                sensor_box = self._create_rounded_box(start_x + 4.5, total_height - 3 - i*0.5, 3, 0.4,
                                                    self.colors['sensors'], sensor)
                ax.add_patch(sensor_box)
                ax.text(start_x + 6, total_height - 2.8 - i*0.5, sensor,
                        ha='center', va='center', fontsize=8, color='white')
        
        # Preprocessing Section
        preprocess_box = self._create_rounded_box(start_x, total_height - 6, 3.5, 1.5,
                                                self.colors['preprocessing'], 'Data Preprocessing')
        ax.add_patch(preprocess_box)
        preprocess_text = 'Data Preprocessing\n• Missing value imputation\n• Zero value correction\n• Outlier removal\n• Standardization'
        ax.text(start_x + 1.75, total_height - 5.25, preprocess_text,
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Physics Feature Engineering Modules
        physics_modules = [
            ('Energy Conservation\nFeatures', 'P_ij = S_i × S_j\n(Power relationships)', self.colors['energy']),
            ('Efficiency Ratio\nFeatures', 'R_ij = S_i/(S_j + ε)\n(Performance ratios)', self.colors['efficiency']),
            ('Statistical Aggregation\nFeatures', 'μ, σ, range, CV\n(System behavior)', self.colors['statistical']),
            ('System-Level\nComposite Features', 'Energy efficiency index\nStability metrics', self.colors['composite'])
        ]
        
        for i, (title, description, color) in enumerate(physics_modules):
            x_pos = start_x + (i % 2) * 4
            y_pos = total_height - 8.5 - (i // 2) * 2
            
            module_box = self._create_rounded_box(x_pos, y_pos, 3.5, 1.5, color, title)
            ax.add_patch(module_box)
            ax.text(x_pos + 1.75, y_pos + 0.75, title,
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            ax.text(x_pos + 1.75, y_pos + 0.25, description,
                    ha='center', va='center', fontsize=7, color='white')
        
        # Enhanced Feature Space Output
        enhanced_box = self._create_rounded_box(start_x + 1, total_height - 12, 6, 1,
                                              self.colors['physics_features'], 'Enhanced Feature Space')
        ax.add_patch(enhanced_box)
        ax.text(start_x + 4, total_height - 11.5, 'Enhanced Feature Space\n(82 physics-informed features)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    def _draw_phase2_classifier_selection(self, ax, start_x, total_height, style):
        """Draw Phase 2: Adaptive Multi-Classifier Selection"""
        
        # Phase 2 Title Box
        phase2_box = FancyBboxPatch(
            (start_x, total_height - 1.5), 8, 1,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['selection'],
            edgecolor='black', linewidth=2
        )
        ax.add_patch(phase2_box)
        ax.text(start_x + 4, total_height - 1, 'PHASE 2: Adaptive Multi-Classifier Selection',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Multi-Algorithm Evaluation
        classifiers = [
            ('Random Forest', 'Ensemble method\nFeature importance'),
            ('AdaBoost', 'Boosting algorithm\nSequential learning'),
            ('Neural Networks', 'Deep learning\nNon-linear patterns'),
            ('Naive Bayes', 'Probabilistic\nIndependence assumption'),
            ('SVM', 'Support vectors\nMargin optimization'),
            ('Logistic Regression', 'Linear classification\nProbabilistic output'),
            ('Gradient Boosting', 'Ensemble boosting\nGradient optimization')
        ]
        
        # Draw classifier evaluation grid
        for i, (name, description) in enumerate(classifiers[:6]):  # Limit to 6 for space
            x_pos = start_x + (i % 3) * 2.5
            y_pos = total_height - 3.5 - (i // 3) * 1.8
            
            classifier_box = self._create_rounded_box(x_pos, y_pos, 2.2, 1.4,
                                                    self.colors['classifiers'], name)
            ax.add_patch(classifier_box)
            ax.text(x_pos + 1.1, y_pos + 0.9, name,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            ax.text(x_pos + 1.1, y_pos + 0.4, description,
                    ha='center', va='center', fontsize=6, color='white')
        
        # Physics Benefit Quantification
        benefit_box = self._create_rounded_box(start_x, total_height - 7.5, 7.5, 1.2,
                                             self.colors['evaluation'], 'Physics Benefit Quantification')
        ax.add_patch(benefit_box)
        benefit_text = 'Physics Benefit Quantification\nΔPerformance = Accuracy_physics - Accuracy_raw\nStatistical significance testing (p < 0.05)'
        ax.text(start_x + 3.75, total_height - 6.9, benefit_text,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Automatic Selection Process
        selection_box = self._create_rounded_box(start_x + 1, total_height - 9.5, 5.5, 1.2,
                                                self.colors['selection'], 'Automatic Selection')
        ax.add_patch(selection_box)
        selection_text = 'Automatic Classifier Selection\nOptimal = max(Physics_Benefit × Absolute_Performance)\nTarget-dependent optimization'
        ax.text(start_x + 3.75, total_height - 8.9, selection_text,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Multi-Target Output
        targets = ['Accumulator\nPressure', 'Cooler\nCondition', 'Pump\nLeakage', 'Valve\nCondition', 'System\nStability']
        for i, target in enumerate(targets):
            target_box = self._create_rounded_box(start_x + i*1.4, total_height - 12, 1.2, 1,
                                                self.colors['output'], target)
            ax.add_patch(target_box)
            ax.text(start_x + i*1.4 + 0.6, total_height - 11.5, target,
                    ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    def _draw_data_flow_connections(self, ax, total_width, total_height):
        """Draw connections showing data flow between phases"""
        
        # Main flow from Phase 1 to Phase 2
        main_arrow = FancyArrowPatch(
            (8.5, total_height - 11.5), (9.5, total_height - 11.5),
            arrowstyle='-|>', mutation_scale=25, 
            color=self.colors['connections'], linewidth=3
        )
        ax.add_patch(main_arrow)
        ax.text(9, total_height - 10.8, 'Enhanced\nFeatures', 
                ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Feedback loop for robustness validation
        feedback_arrow = FancyArrowPatch(
            (total_width/2 + 4, total_height - 8), (6, total_height - 8),
            arrowstyle='-|>', mutation_scale=20,
            color=self.colors['evaluation'], linewidth=2, linestyle='--'
        )
        ax.add_patch(feedback_arrow)
        ax.text(total_width/2 + 1, total_height - 7.5, 'Robustness\nFeedback',
                ha='center', va='center', fontsize=7, style='italic')
    
    def _draw_evaluation_metrics_section(self, ax, start_x, start_y):
        """Draw evaluation metrics section"""
        
        metrics_box = self._create_rounded_box(start_x, start_y, 7, 2,
                                             self.colors['evaluation'], 'Evaluation Metrics')
        ax.add_patch(metrics_box)
        
        metrics_text = '''Comprehensive Validation Framework
• 10-fold Cross-validation with stratification
• Statistical significance testing (p < 0.05)
• Ablation resistance analysis (1.20× improvement)
• Permutation importance validation
• Multi-target performance assessment'''
        
        ax.text(start_x + 3.5, start_y + 1, metrics_text,
                ha='center', va='center', fontsize=8, color='white')
    
    def _create_rounded_box(self, x, y, width, height, color, label=None):
        """Create a rounded rectangle box"""
        return FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.9,
            edgecolor='black', linewidth=1
        )
    
    def _add_title_and_metadata(self, ax, total_width, total_height, style):
        """Add title and framework metadata"""
        
        # Main title
        ax.text(total_width/2, total_height + 0.5, 
                'PEECOM Framework Architecture',
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Subtitle
        ax.text(total_width/2, total_height + 0.1,
                'Predictive Energy Efficiency Control and Optimization Model',
                ha='center', va='center', fontsize=12, style='italic')
        
        # Framework statistics
        stats_text = '''Framework Statistics: 46 → 82 features | 7 classifiers | 5 targets | 97-99% accuracy
Robustness: 1.20× ablation resistance | Statistical significance: p < 0.001'''
        
        ax.text(total_width/2, -0.5, stats_text,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    def create_simplified_graphviz_diagram(self):
        """Create a simplified PEECOM diagram using Graphviz"""
        
        if not HAS_GRAPHVIZ:
            print("⚠️ Graphviz not available. Skipping simplified diagram.")
            return None
        
        # Initialize diagram
        diagram = Digraph("PEECOM_Framework", format="png")
        diagram.attr(rankdir="LR", splines="ortho", nodesep="0.8", ranksep="1.2")
        diagram.attr('node', style='filled', fontname='Arial')
        
        # Input Data
        diagram.node("input", "Raw Sensor Data\n(46 sensors)\n\nPS1-PS6, TS1-TS4\nFS1-FS2, EPS1\nCE, CP, SE",
                    shape="box", fillcolor="#2E86AB", fontcolor="white")
        
        # Phase 1: Physics-Informed Feature Engineering
        with diagram.subgraph(name='cluster_phase1') as phase1:
            phase1.attr(label='Phase 1: Physics-Informed Feature Engineering',
                        style='filled', fillcolor='#F0F8FF', fontsize='14')
            
            # Preprocessing
            phase1.node("preprocess", "Data Preprocessing\n\n• Missing value imputation\n• Zero correction\n• Outlier removal\n• Standardization",
                       shape="box", fillcolor="#C73E1D", fontcolor="white")
            
            # Physics feature modules
            phase1.node("energy", "Energy Conservation\nFeatures\n\nP_ij = S_i × S_j",
                       shape="ellipse", fillcolor="#E76F51", fontcolor="white")
            phase1.node("efficiency", "Efficiency Ratio\nFeatures\n\nR_ij = S_i/(S_j + ε)",
                       shape="ellipse", fillcolor="#F4A261", fontcolor="white")
            phase1.node("statistical", "Statistical Aggregation\nFeatures\n\nμ, σ, range, CV",
                       shape="ellipse", fillcolor="#E9C46A", fontcolor="white")
            phase1.node("composite", "System-Level\nComposite Features\n\nStability metrics",
                       shape="ellipse", fillcolor="#2A9D8F", fontcolor="white")
            
            # Enhanced features
            phase1.node("enhanced", "Enhanced Feature Space\n(82 physics-informed features)",
                       shape="box", fillcolor="#F18F01", fontcolor="white")
        
        # Phase 2: Adaptive Multi-Classifier Selection
        with diagram.subgraph(name='cluster_phase2') as phase2:
            phase2.attr(label='Phase 2: Adaptive Multi-Classifier Selection',
                        style='filled', fillcolor='#F5F0FF', fontsize='14')
            
            # Multi-classifier evaluation
            phase2.node("classifiers", "Multi-Algorithm Evaluation\n\n• Random Forest\n• AdaBoost\n• Neural Networks\n• Naive Bayes\n• SVM\n• Logistic Regression\n• Gradient Boosting",
                       shape="diamond", fillcolor="#264653", fontcolor="white")
            
            # Physics benefit quantification
            phase2.node("benefit", "Physics Benefit\nQuantification\n\nΔ = Acc_physics - Acc_raw\nStatistical testing",
                       shape="hexagon", fillcolor="#5D737E", fontcolor="white")
            
            # Automatic selection
            phase2.node("selection", "Automatic Selection\n\nOptimal classifier\nper target",
                       shape="diamond", fillcolor="#6A4C93", fontcolor="white")
        
        # Output targets
        diagram.node("output", "Multi-Target Output\n\n• Accumulator Pressure\n• Cooler Condition\n• Pump Leakage\n• Valve Condition\n• System Stability",
                    shape="box", fillcolor="#679436", fontcolor="white")
        
        # Evaluation
        diagram.node("evaluation", "Comprehensive Validation\n\n• 10-fold Cross-validation\n• Statistical significance\n• Ablation resistance (1.20×)\n• Permutation importance",
                    shape="box", fillcolor="#8B5A3C", fontcolor="white")
        
        # Add edges
        diagram.edge("input", "preprocess", label="Raw data")
        diagram.edge("preprocess", "energy")
        diagram.edge("preprocess", "efficiency")
        diagram.edge("preprocess", "statistical")
        diagram.edge("preprocess", "composite")
        diagram.edge("energy", "enhanced")
        diagram.edge("efficiency", "enhanced")
        diagram.edge("statistical", "enhanced")
        diagram.edge("composite", "enhanced")
        diagram.edge("enhanced", "classifiers", label="82 features")
        diagram.edge("classifiers", "benefit")
        diagram.edge("benefit", "selection")
        diagram.edge("selection", "output", label="Optimal model")
        diagram.edge("output", "evaluation")
        diagram.edge("evaluation", "benefit", style="dashed", label="Feedback")
        
        # Render diagram
        output_path = self.output_dir / "peecom_framework_simplified"
        diagram.render(str(output_path), cleanup=True)
        
        print(f"✅ Simplified PEECOM framework diagram saved: {output_path}.png")
        return f"{output_path}.png"
    
    def create_technical_architecture_diagram(self):
        """Create a technical architecture diagram showing implementation details"""
        
        fig, ax = plt.subplots(figsize=(18, 12))
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        # Title
        ax.text(9, 11.5, 'PEECOM Technical Architecture', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Data Layer
        data_components = [
            ('CMOHS Dataset\n2,205 samples\n68 features', 1, 10),
            ('MotorVD Dataset\n107,346 samples\nCross-validation', 5, 10),
            ('Preprocessing\nPipeline', 9, 10),
            ('Feature\nStandardization', 13, 10)
        ]
        
        for name, x, y in data_components:
            box = self._create_rounded_box(x, y, 3, 1.2, self.colors['input'], name)
            ax.add_patch(box)
            ax.text(x + 1.5, y + 0.6, name, ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        # Model Layer
        model_components = [
            ('Simple PEECOM\nBaseline model\nRandom Forest', 1, 7.5),
            ('Multi-Classifier\nPEECOM\n7 algorithms', 5, 7.5),
            ('Enhanced PEECOM\nAdvanced features\nOptimized selection', 9, 7.5),
            ('Validation Suite\nRobustness testing\nStatistical analysis', 13, 7.5)
        ]
        
        for name, x, y in model_components:
            box = self._create_rounded_box(x, y, 3, 1.5, self.colors['classifiers'], name)
            ax.add_patch(box)
            ax.text(x + 1.5, y + 0.75, name, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
        
        # Output Layer
        output_components = [
            ('Performance\nMetrics\nF1, Precision, Recall', 2, 5),
            ('Feature Importance\nPermutation analysis\nPhysics validation', 6, 5),
            ('Robustness\nAssessment\nAblation resistance', 10, 5),
            ('Publication\nPlots\n6 figures generated', 14, 5)
        ]
        
        for name, x, y in output_components:
            box = self._create_rounded_box(x, y, 3, 1.5, self.colors['output'], name)
            ax.add_patch(box)
            ax.text(x + 1.5, y + 0.75, name, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
        
        # Implementation Details
        impl_box = self._create_rounded_box(2, 2, 14, 2, self.colors['evaluation'], 'Implementation')
        ax.add_patch(impl_box)
        impl_text = '''Implementation Framework
• Python-based modular architecture with src/models/, src/loader/, src/utils/, src/visualization/
• Scikit-learn integration for machine learning algorithms and preprocessing pipelines
• Comprehensive validation using stratified k-fold cross-validation and statistical significance testing
• Automated model selection with physics benefit quantification and robustness assessment
• Publication-ready visualization system with 6 comprehensive figures and Excel exports for reviewers'''
        
        ax.text(9, 3, impl_text, ha='center', va='center', fontsize=9, color='white')
        
        # Add connections
        for i in range(4):
            # Data to Model connections
            arrow1 = FancyArrowPatch((2.5 + i*4, 9.8), (2.5 + i*4, 9.2),
                                   arrowstyle='->', mutation_scale=15,
                                   color=self.colors['connections'], linewidth=2)
            ax.add_patch(arrow1)
            
            # Model to Output connections  
            arrow2 = FancyArrowPatch((2.5 + i*4, 7.3), (2.5 + i*4, 6.7),
                                   arrowstyle='->', mutation_scale=15,
                                   color=self.colors['connections'], linewidth=2)
            ax.add_patch(arrow2)
        
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Save diagram
        filename = "peecom_technical_architecture.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        print(f"✅ Technical architecture diagram saved: {filepath}")
        return filepath
    
    def generate_all_diagrams(self, style='detailed'):
        """Generate all PEECOM framework diagrams"""
        
        print("🎨 Generating PEECOM Framework Schematic Diagrams...")
        print("=" * 60)
        
        generated_files = []
        
        # 1. Comprehensive framework diagram
        print("📊 Creating comprehensive framework diagram...")
        comprehensive_file = self.create_comprehensive_framework_diagram(style)
        generated_files.append(comprehensive_file)
        
        # 2. Simplified Graphviz diagram
        if HAS_GRAPHVIZ:
            print("📈 Creating simplified Graphviz diagram...")
            graphviz_file = self.create_simplified_graphviz_diagram()
            if graphviz_file:
                generated_files.append(graphviz_file)
        
        # 3. Technical architecture diagram
        print("🔧 Creating technical architecture diagram...")
        technical_file = self.create_technical_architecture_diagram()
        generated_files.append(technical_file)
        
        print("\n" + "=" * 60)
        print("✅ PEECOM Framework Visualization Complete!")
        print("=" * 60)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📄 Generated {len(generated_files)} diagram files:")
        for file in generated_files:
            if file:
                print(f"   • {Path(file).name}")
        
        return generated_files


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='PEECOM Framework Schematic Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python peecom_framework_visualizer.py
  python peecom_framework_visualizer.py --style simplified
  python peecom_framework_visualizer.py --output output/diagrams --style detailed
        '''
    )
    
    parser.add_argument('--output', type=str, default='output/figures',
                       help='Output directory for generated diagrams (default: output/figures)')
    parser.add_argument('--style', choices=['detailed', 'simplified'], default='detailed',
                       help='Diagram style: detailed (default) or simplified')
    parser.add_argument('--format', choices=['png', 'svg', 'pdf'], default='png',
                       help='Output format for diagrams (default: png)')
    
    return parser.parse_args()


def main():
    """Main function to generate PEECOM framework diagrams"""
    
    args = parse_arguments()
    
    print("🎯 PEECOM Framework Schematic Generator")
    print("=" * 50)
    print(f"Output directory: {args.output}")
    print(f"Style: {args.style}")
    print(f"Format: {args.format}")
    print("=" * 50)
    
    # Create visualizer
    visualizer = PEECOMFrameworkVisualizer(output_dir=args.output)
    
    # Generate all diagrams
    generated_files = visualizer.generate_all_diagrams(style=args.style)
    
    print(f"\n🎉 Successfully generated {len(generated_files)} PEECOM framework diagrams!")
    
    return 0


if __name__ == "__main__":
    exit(main())