"""
PEECOM & BLAST Framework: Experimental Design Architecture Flowchart
Shows the dual-role experimental design with diagnostic vs application models
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_dual_role_experimental_design():
    """Create flowchart showing dual-role experimental design"""
    
    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    colors = {
        'blast': '#FF6B6B',      # Red for BLAST
        'diagnostic': '#4ECDC4',  # Teal for diagnostics
        'application': '#45B7D1', # Blue for application models
        'data': '#DDA0DD',        # Plum for data
        'validation': '#54A0FF',  # Light blue for validation
        'results': '#96CEB4'      # Green for results
    }
    
    # Title
    ax.text(11, 13.5, 'PEECOM & BLAST: Dual-Role Experimental Design Architecture', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(11, 13, 'Diagnostic Tools vs Protected Application Models', 
            ha='center', va='center', fontsize=16, style='italic')
    
    # Central Data Source
    _add_rounded_box(ax, 9, 11, 4, 1.5, "CMOHS Dataset\n2,205 samples × 54 features\n3 temporal blocks\nBalanced class distribution", 
                     colors['data'], 'black')
    
    # Left Side: DIAGNOSTIC TOOLS (BLAST Components)
    ax.text(5.5, 10, 'DIAGNOSTIC TOOLS\n(BLAST Framework)', 
            ha='center', va='center', fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['blast'], alpha=0.3))
    
    _add_rounded_box(ax, 1, 8.5, 4.5, 1.2, "RandomForest Block Predictor\nRole: Leakage Detection\nTask: Predict data collection blocks\nPurpose: Quality control diagnostic", 
                     colors['diagnostic'], 'white')
    
    _add_rounded_box(ax, 1, 6.5, 4.5, 1.2, "Feature Fingerprinting Analysis\nRole: Root cause identification\nTask: Cohen's d effect sizes\nPurpose: Identify problematic sensors", 
                     colors['diagnostic'], 'white')
    
    _add_rounded_box(ax, 1, 4.5, 4.5, 1.2, "BLAST Remediation Engine\nRole: Data sanitization\nTask: Block normalization\nPurpose: Eliminate systematic bias", 
                     colors['blast'], 'white')
    
    # Right Side: APPLICATION MODELS (Protected Testbeds)
    ax.text(16.5, 10, 'APPLICATION MODELS\n(Protected Testbeds)', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['application'], alpha=0.3))
    
    _add_rounded_box(ax, 16.5, 8.5, 4.5, 1.2, "Simple PEECOM Testbed\nRole: Baseline application model\nTask: Hydraulic condition classification\nPurpose: Demonstrate vulnerability", 
                     colors['application'], 'white')
    
    _add_rounded_box(ax, 16.5, 6.5, 4.5, 1.2, "Enhanced PEECOM Testbed\nRole: Production-grade model\nTask: Physics-informed prediction\nPurpose: Show universal susceptibility", 
                     colors['application'], 'white')
    
    _add_rounded_box(ax, 16.5, 4.5, 4.5, 1.2, "Additional ML Testbeds\nRole: Cross-architecture validation\nTask: Various classification approaches\nPurpose: Framework generalizability", 
                     colors['application'], 'white')
    
    # Central Validation Protocol
    _add_rounded_box(ax, 8, 2.5, 6, 1.5, "Unified Validation Protocol\n• Multi-seed cross-validation (42, 123, 456)\n• Permutation testing (1,000+ iterations)\n• Statistical significance (p-values)\n• Effect size quantification (Cohen's d)", 
                     colors['validation'], 'white')
    
    # Bottom Results
    _add_rounded_box(ax, 2, 0.5, 8, 1.2, "DIAGNOSTIC RESULTS:\n• Block prediction: 95.8% ± 2.1% → 33.3% ± 0.2%\n• Feature analysis: Cohen's d > 3.7 → ≈ 0\n• Remediation: Complete leakage elimination", 
                     colors['diagnostic'], 'white')
    
    _add_rounded_box(ax, 12, 0.5, 8, 1.2, "APPLICATION RESULTS:\n• Simple PEECOM: High accuracy → 33.2% ± 0.6%\n• Enhanced PEECOM: Similar exploitation → 33.1% ± 0.3%\n• All models: Protected from block artifacts", 
                     colors['application'], 'white')
    
    # Key Insights Box
    ax.text(11, 12.2, '🔑 KEY INSIGHT: Dual-Role Design Separates Concerns', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Add arrows showing data flow
    # From central data to diagnostic tools
    _add_arrow(ax, 9, 11.5, 5.5, 9.7)  # To RandomForest
    _add_arrow(ax, 9, 11.3, 3.25, 7.7)  # To Feature Analysis
    _add_arrow(ax, 9, 11.1, 3.25, 5.7)  # To Remediation
    
    # From central data to application models  
    _add_arrow(ax, 13, 11.5, 16.5, 9.7)  # To Simple PEECOM
    _add_arrow(ax, 13, 11.3, 18.75, 7.7)  # To Enhanced PEECOM
    _add_arrow(ax, 13, 11.1, 18.75, 5.7)  # To Additional testbeds
    
    # To validation protocol
    _add_arrow(ax, 5.5, 4.5, 8, 3.7)  # From diagnostics
    _add_arrow(ax, 16.5, 4.5, 14, 3.7)  # From applications
    
    # To results
    _add_arrow(ax, 9, 2.5, 6, 1.7)  # To diagnostic results
    _add_arrow(ax, 13, 2.5, 16, 1.7)  # To application results
    
    # Add explanatory text boxes
    ax.text(5.5, 7.5, 'These tools DETECT\nand REMEDIATE\nleakage patterns', 
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.5))
    
    ax.text(16.5, 7.5, 'These models are\nPROTECTED by\nBLAST remediation', 
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.5))
    
    plt.title("Dual-Role Experimental Design: Diagnostic Tools vs Protected Application Models", 
             fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('PEECOM_BLAST_Dual_Role_Experimental_Design.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def _add_rounded_box(ax, x, y, width, height, text, facecolor, textcolor):
    """Add a rounded rectangle with text"""
    box = FancyBboxPatch((x, y), width, height,
                       boxstyle="round,pad=0.1",
                       facecolor=facecolor,
                       edgecolor='black',
                       linewidth=1.5)
    ax.add_patch(box)
    
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=textcolor,
            wrap=True)

def _add_arrow(ax, x1, y1, x2, y2):
    """Add an arrow between two points"""
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle='->', 
                                  mutation_scale=20,
                                  color='black',
                                  linewidth=2)
    ax.add_patch(arrow)

if __name__ == "__main__":
    print("🎨 Generating Dual-Role Experimental Design Flowchart...")
    create_dual_role_experimental_design()
    print("✅ PEECOM_BLAST_Dual_Role_Experimental_Design.png created successfully!")