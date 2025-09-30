"""
PEECOM/BLAST Framework Flowcharts
Generate comprehensive visual flowcharts for methodological protocol with multiple testbeds
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

class FrameworkFlowchartGenerator:
    
    def __init__(self):
        self.colors = {
            'blast': '#FF6B6B',      # Red for BLAST
            'diagnostic': '#4ECDC4',  # Teal for diagnostics
            'simple_peecom': '#45B7D1', # Blue for Simple PEECOM
            'enhanced_peecom': '#96CEB4', # Green for Enhanced PEECOM
            'random_forest': '#FECA57', # Yellow for RandomForest
            'remediation': '#FF9FF3',  # Pink for remediation
            'validation': '#54A0FF',   # Light blue for validation
            'data': '#DDA0DD'         # Plum for data
        }
    
    def create_comprehensive_methodology_flowchart(self):
        """Create comprehensive flowchart showing entire methodology with all testbeds"""
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Title
        ax.text(10, 15.5, 'PEECOM & BLAST: Comprehensive Framework for Block-Level Data Leakage Detection', 
                ha='center', va='center', fontsize=18, fontweight='bold')
        ax.text(10, 15, 'Multi-Testbed Methodological Protocol with Diagnostic Cascade', 
                ha='center', va='center', fontsize=14, style='italic')
        
        # Phase 1: Data Input and Initial Setup
        self._add_rounded_box(ax, 1, 13.5, 3, 1, "CMOHS Dataset\n2,205 samples\n54 features\n3 temporal blocks", 
                             self.colors['data'], 'white')
        
        # Phase 2: BLAST Diagnostic Cascade
        self._add_rounded_box(ax, 6, 13.5, 3.5, 1, "BLAST Diagnostic\nRandomForest Block Predictor\nDetect Systematic Leakage", 
                             self.colors['blast'], 'white')
        
        # Phase 3: Feature Analysis
        self._add_rounded_box(ax, 11, 13.5, 3.5, 1, "Feature Fingerprinting\nCohen's d Analysis\nTop Predictive Features", 
                             self.colors['diagnostic'], 'white')
        
        # Phase 4: Multiple Testbed Architecture
        # RandomForest Diagnostic Testbed
        self._add_rounded_box(ax, 1, 11, 4, 1.2, "RandomForest Diagnostic\nTestbed\n• Block prediction task\n• Leakage quantification\n• Effect size analysis", 
                             self.colors['random_forest'], 'black')
        
        # Simple PEECOM Testbed  
        self._add_rounded_box(ax, 6.5, 11, 4, 1.2, "Simple PEECOM Testbed\n• Standard RandomForest\n• Basic feature engineering\n• Statistical aggregations\n• Baseline vulnerability", 
                             self.colors['simple_peecom'], 'white')
        
        # Enhanced PEECOM Testbed
        self._add_rounded_box(ax, 12, 11, 4, 1.2, "Enhanced PEECOM Testbed\n• Physics-informed features\n• Energy domain analysis\n• Thermodynamic relations\n• Production-grade pipeline", 
                             self.colors['enhanced_peecom'], 'black')
        
        # Phase 5: Leakage Detection Results
        self._add_rounded_box(ax, 1, 8.5, 2.5, 1, "RandomForest:\n95.8% ± 2.1%\nBlock Prediction\nSEVERE LEAKAGE", 
                             self.colors['random_forest'], 'black')
        
        self._add_rounded_box(ax, 4.5, 8.5, 2.5, 1, "Simple PEECOM:\nHigh Accuracy\nLeakage Exploitation\nBaseline Vulnerability", 
                             self.colors['simple_peecom'], 'white')
        
        self._add_rounded_box(ax, 8, 8.5, 2.5, 1, "Enhanced PEECOM:\nSimilar Exploitation\nSophisticated Yet\nVulnerable", 
                             self.colors['enhanced_peecom'], 'black')
        
        # Phase 6: BLAST Remediation Framework
        self._add_rounded_box(ax, 12, 8.5, 4, 1, "BLAST Remediation\n• Block mean normalization\n• Covariance alignment\n• Comprehensive sanitization", 
                             self.colors['remediation'], 'black')
        
        # Phase 7: Post-Remediation Validation
        self._add_rounded_box(ax, 3, 6, 3.5, 1, "Multi-Seed Cross-Validation\nSeeds: 42, 123, 456\nStratifiedKFold (5-fold)\nRobustness Testing", 
                             self.colors['validation'], 'white')
        
        self._add_rounded_box(ax, 8, 6, 3.5, 1, "Permutation Testing\n1,000+ iterations\nStatistical Significance\nNull Hypothesis Testing", 
                             self.colors['validation'], 'white')
        
        self._add_rounded_box(ax, 13, 6, 3.5, 1, "Effect Size Analysis\nCohen's d Quantification\nPractical Significance\nLiterature Standards", 
                             self.colors['validation'], 'white')
        
        # Phase 8: Final Results
        self._add_rounded_box(ax, 1, 3.5, 5, 1.5, "Remediation Success:\n• RandomForest: 33.3% ± 0.2% (chance level)\n• Simple PEECOM: 33.2% ± 0.6%\n• Enhanced PEECOM: 33.1% ± 0.3%\n• All p-values > 0.05\n• Effect sizes ≈ 0", 
                             self.colors['validation'], 'white')
        
        self._add_rounded_box(ax, 8, 3.5, 5, 1.5, "Universal Framework Value:\n• Methodological breakthrough\n• Cross-domain applicability\n• Medical devices, IoT, automotive\n• Industrial monitoring systems\n• Reliable ML deployment", 
                             self.colors['blast'], 'white')
        
        # Phase 9: Publication Impact
        self._add_rounded_box(ax, 14.5, 3.5, 4.5, 1.5, "Research Contributions:\n• Dual-artifact framework\n• PEECOM: Application testbed\n• BLAST: Universal toolkit\n• Methodological rigor\n• False discovery prevention", 
                             self.colors['diagnostic'], 'white')
        
        # Add arrows connecting the flow
        self._add_arrow(ax, 2.5, 13.5, 5.5, 13.5)  # Data → BLAST
        self._add_arrow(ax, 7.75, 13.5, 10.5, 13.5)  # BLAST → Features
        
        # From initial setup to testbeds
        self._add_arrow(ax, 3, 12.8, 3, 12.2)  # To RandomForest
        self._add_arrow(ax, 6, 12.8, 8.5, 12.2)  # To Simple PEECOM
        self._add_arrow(ax, 12, 12.8, 14, 12.2)  # To Enhanced PEECOM
        
        # From testbeds to results
        self._add_arrow(ax, 3, 10.5, 2.25, 9.5)  # RandomForest results
        self._add_arrow(ax, 8.5, 10.5, 5.75, 9.5)  # Simple PEECOM results
        self._add_arrow(ax, 14, 10.5, 9.25, 9.5)  # Enhanced PEECOM results
        
        # To remediation
        self._add_arrow(ax, 11, 8.5, 12, 8.5)  # To BLAST remediation
        
        # To validation phases
        self._add_arrow(ax, 14, 8, 4.75, 7)  # To cross-validation
        self._add_arrow(ax, 14, 8, 9.75, 7)  # To permutation testing
        self._add_arrow(ax, 14, 8, 14.75, 7)  # To effect size
        
        # To final results
        self._add_arrow(ax, 4.75, 6, 3.5, 5)  # To success metrics
        self._add_arrow(ax, 9.75, 6, 10.5, 5)  # To framework value
        self._add_arrow(ax, 14.75, 6, 16.75, 5)  # To research impact
        
        plt.title("Comprehensive PEECOM & BLAST Framework: Multi-Testbed Methodological Protocol", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('PEECOM_BLAST_Comprehensive_Methodology_Flowchart.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_testbed_comparison_flowchart(self):
        """Create side-by-side comparison of all testbeds"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 12))
        
        # RandomForest Diagnostic Testbed
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 12)
        ax1.axis('off')
        ax1.set_title('RandomForest Diagnostic Testbed\n(BLAST Component)', 
                     fontsize=14, fontweight='bold', color=self.colors['random_forest'])
        
        # Input
        self._add_rounded_box(ax1, 2, 10.5, 6, 1, "CMOHS Dataset\nX: 2,205 × 54 features\nblock_labels: [0,1,2]", 
                             self.colors['data'], 'black')
        
        # Architecture
        self._add_rounded_box(ax1, 1, 8.5, 8, 1.2, "RandomForest Architecture:\n• 100 estimators, max_depth=10\n• Block prediction task\n• Diagnostic purpose only", 
                             self.colors['random_forest'], 'black')
        
        # Process
        self._add_rounded_box(ax1, 1, 6.5, 8, 1.2, "Diagnostic Process:\n• StratifiedKFold CV (5-fold)\n• Predict data collection blocks\n• Quantify systematic artifacts", 
                             self.colors['diagnostic'], 'white')
        
        # Results
        self._add_rounded_box(ax1, 1, 4.5, 8, 1.2, "Leakage Detection Results:\n• 95.8% ± 2.1% accuracy\n• Chance level: 33.3%\n• SEVERE LEAKAGE DETECTED", 
                             'red', 'white')
        
        # Purpose
        self._add_rounded_box(ax1, 1, 2.5, 8, 1.2, "Framework Role:\n• Universal leakage detector\n• Domain-agnostic tool\n• Quality control standard", 
                             self.colors['blast'], 'white')
        
        # Simple PEECOM Testbed
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 12)
        ax2.axis('off')
        ax2.set_title('Simple PEECOM Testbed\n(Application Model - Baseline)', 
                     fontsize=14, fontweight='bold', color=self.colors['simple_peecom'])
        
        # Input
        self._add_rounded_box(ax2, 2, 10.5, 6, 1, "Original Features\n+ Basic Engineering\nStatistical Aggregations", 
                             self.colors['data'], 'black')
        
        # Architecture
        self._add_rounded_box(ax2, 1, 8.5, 8, 1.2, "Simple PEECOM Pipeline:\n• StandardScaler normalization\n• RandomForest classifier\n• Basic feature expansion", 
                             self.colors['simple_peecom'], 'white')
        
        # Features
        self._add_rounded_box(ax2, 1, 6.5, 8, 1.2, "Feature Engineering:\n• Raw sensors (54)\n• Mean, std, min, max\n• Total: 58 features", 
                             self.colors['diagnostic'], 'white')
        
        # Results  
        self._add_rounded_box(ax2, 1, 4.5, 8, 1.2, "Performance Results:\n• High accuracy pre-remediation\n• Exploits block artifacts\n• Baseline vulnerability", 
                             'orange', 'black')
        
        # Purpose
        self._add_rounded_box(ax2, 1, 2.5, 8, 1.2, "Testbed Role:\n• Baseline application model\n• Standard ML pipeline\n• Leakage susceptibility demo", 
                             self.colors['simple_peecom'], 'white')
        
        # Enhanced PEECOM Testbed
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 12)
        ax3.axis('off')
        ax3.set_title('Enhanced PEECOM Testbed\n(Application Model - Production)', 
                     fontsize=14, fontweight='bold', color=self.colors['enhanced_peecom'])
        
        # Input
        self._add_rounded_box(ax3, 2, 10.5, 6, 1, "Physics-Informed\nFeature Engineering\nThermodynamic Relations", 
                             self.colors['data'], 'black')
        
        # Architecture
        self._add_rounded_box(ax3, 1, 8.5, 8, 1.2, "Enhanced PEECOM Pipeline:\n• Advanced preprocessing\n• Physics-informed features\n• Production-grade complexity", 
                             self.colors['enhanced_peecom'], 'black')
        
        # Features
        self._add_rounded_box(ax3, 1, 6.5, 8, 1.2, "Advanced Features:\n• Energy domain aggregations\n• Thermodynamic relationships\n• System efficiency metrics", 
                             self.colors['diagnostic'], 'white')
        
        # Results
        self._add_rounded_box(ax3, 1, 4.5, 8, 1.2, "Performance Results:\n• Similar leakage exploitation\n• Sophistication ≠ protection\n• Validates framework necessity", 
                             'orange', 'black')
        
        # Purpose
        self._add_rounded_box(ax3, 1, 2.5, 8, 1.2, "Testbed Role:\n• Production-grade model\n• Demonstrates universality\n• Real-world applicability", 
                             self.colors['enhanced_peecom'], 'black')
        
        # Add arrows for each testbed
        for ax in [ax1, ax2, ax3]:
            self._add_arrow(ax, 5, 10, 5, 9.7)  # Input to architecture
            self._add_arrow(ax, 5, 8, 5, 7.7)   # Architecture to process/features
            self._add_arrow(ax, 5, 6, 5, 5.7)   # Process to results
            self._add_arrow(ax, 5, 4, 5, 3.7)   # Results to purpose
        
        plt.suptitle('PEECOM & BLAST: Multi-Testbed Experimental Design', 
                    fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig('PEECOM_BLAST_Testbed_Comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_remediation_validation_flowchart(self):
        """Create flowchart showing remediation and validation process"""
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        ax.text(9, 13.5, 'BLAST Remediation & Multi-Testbed Validation Protocol', 
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Phase 1: Pre-remediation state
        self._add_rounded_box(ax, 1, 11.5, 4.5, 1.5, "Pre-Remediation State:\n• RandomForest: 95.8% ± 2.1%\n• Simple PEECOM: High accuracy\n• Enhanced PEECOM: Similar\n• All exploit block artifacts", 
                             'red', 'white')
        
        # Phase 2: BLAST Remediation
        self._add_rounded_box(ax, 7, 11.5, 4, 1.5, "BLAST Remediation:\n• Block mean normalization\n• Covariance alignment\n• Systematic difference removal", 
                             self.colors['remediation'], 'black')
        
        # Phase 3: Multi-testbed validation
        self._add_rounded_box(ax, 12.5, 11.5, 4.5, 1.5, "Multi-Testbed Validation:\n• All testbeds re-evaluated\n• Same architectures\n• Remediated data only", 
                             self.colors['validation'], 'white')
        
        # Validation components
        self._add_rounded_box(ax, 1, 9, 5, 1, "Multi-Seed Cross-Validation\nSeeds: [42, 123, 456]\nStratifiedKFold (n_splits=5)", 
                             self.colors['validation'], 'white')
        
        self._add_rounded_box(ax, 7, 9, 5, 1, "Permutation Testing\n1,000+ iterations per testbed\nNull hypothesis validation", 
                             self.colors['validation'], 'white')
        
        self._add_rounded_box(ax, 13, 9, 4, 1, "Effect Size Analysis\nCohen's d quantification\nPractical significance", 
                             self.colors['validation'], 'white')
        
        # Results for each testbed
        self._add_rounded_box(ax, 0.5, 6.5, 5, 1.5, "RandomForest Results:\n• 33.3% ± 0.2% accuracy\n• p-values: 0.501, 0.409, 0.506\n• Effect sizes ≈ 0\n• Chance-level performance", 
                             self.colors['random_forest'], 'black')
        
        self._add_rounded_box(ax, 6.5, 6.5, 5, 1.5, "Simple PEECOM Results:\n• 33.2% ± 0.6% accuracy\n• Statistical insignificance\n• Leakage eliminated\n• Baseline protection achieved", 
                             self.colors['simple_peecom'], 'white')
        
        self._add_rounded_box(ax, 12.5, 6.5, 5, 1.5, "Enhanced PEECOM Results:\n• 33.1% ± 0.3% accuracy\n• Same remediation success\n• Production model protected\n• Framework scalability proven", 
                             self.colors['enhanced_peecom'], 'black')
        
        # Success criteria
        self._add_rounded_box(ax, 2, 4, 6, 1.5, "Remediation Success Criteria:\n• Accuracy = 33.3% ± 0.2% (chance)\n• p-values > 0.05 (insignificant)\n• |Cohen's d| < 0.1 (negligible)\n• Consistent across all seeds", 
                             'green', 'white')
        
        # Framework validation
        self._add_rounded_box(ax, 10, 4, 6, 1.5, "Framework Validation:\n• Universal remediation success\n• Cross-testbed consistency\n• Methodological robustness\n• Deployment readiness", 
                             self.colors['blast'], 'white')
        
        # Universal impact
        self._add_rounded_box(ax, 4, 1.5, 10, 1.5, "Universal Research Impact:\n• Medical devices, IoT sensors, autonomous vehicles\n• Environmental monitoring, industrial automation\n• Prevents false discoveries across domains\n• Establishes new validation standards", 
                             self.colors['diagnostic'], 'white')
        
        # Add connecting arrows
        self._add_arrow(ax, 5.5, 11.5, 7, 11.5)  # Pre to remediation
        self._add_arrow(ax, 11, 11.5, 12.5, 11.5)  # Remediation to validation
        
        # To validation components
        self._add_arrow(ax, 9, 11, 3.5, 10)  # To multi-seed
        self._add_arrow(ax, 9, 11, 9.5, 10)  # To permutation
        self._add_arrow(ax, 9, 11, 15, 10)  # To effect size
        
        # To testbed results
        self._add_arrow(ax, 3.5, 9, 3, 8)  # Multi-seed to results
        self._add_arrow(ax, 9.5, 9, 9, 8)  # Permutation to results
        self._add_arrow(ax, 15, 9, 15, 8)  # Effect size to results
        
        # To success criteria and validation
        self._add_arrow(ax, 5, 6.5, 5, 5.5)  # To success criteria
        self._add_arrow(ax, 13, 6.5, 13, 5.5)  # To framework validation
        
        # To universal impact
        self._add_arrow(ax, 8, 4, 9, 3)  # To universal impact
        
        plt.title("BLAST Remediation & Multi-Testbed Validation Protocol", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('PEECOM_BLAST_Remediation_Validation_Flowchart.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def _add_rounded_box(self, ax, x, y, width, height, text, facecolor, textcolor):
        """Add a rounded rectangle with text"""
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.1",
                           facecolor=facecolor,
                           edgecolor='black',
                           linewidth=1.5)
        ax.add_patch(box)
        
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=textcolor,
                wrap=True)
    
    def _add_arrow(self, ax, x1, y1, x2, y2):
        """Add an arrow between two points"""
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle='->', 
                                      mutation_scale=20,
                                      color='black',
                                      linewidth=2)
        ax.add_patch(arrow)

# Generate all flowcharts
if __name__ == "__main__":
    generator = FrameworkFlowchartGenerator()
    
    print("🎨 Generating PEECOM & BLAST Framework Flowcharts...")
    
    # Generate comprehensive methodology flowchart
    print("📊 Creating comprehensive methodology flowchart...")
    generator.create_comprehensive_methodology_flowchart()
    
    # Generate testbed comparison flowchart
    print("🔬 Creating testbed comparison flowchart...")
    generator.create_testbed_comparison_flowchart()
    
    # Generate remediation and validation flowchart
    print("✅ Creating remediation validation flowchart...")
    generator.create_remediation_validation_flowchart()
    
    print("\n🎯 All flowcharts generated successfully!")
    print("Files created:")
    print("  1. PEECOM_BLAST_Comprehensive_Methodology_Flowchart.png")
    print("  2. PEECOM_BLAST_Testbed_Comparison.png")
    print("  3. PEECOM_BLAST_Remediation_Validation_Flowchart.png")