"""
Simplified PEECOM Framework Core Methodology Flowchart
Shows the essential flow between major components
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_core_methodology_flowchart():
    """Create simplified flowchart showing core methodology flow"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = {
        'data': '#DDA0DD',        # Purple for data
        'diagnostic': '#FF6B6B',  # Red for diagnostic tools
        'peecom': '#45B7D1',      # Blue for PEECOM
        'remediation': '#96CEB4', # Green for remediation
        'validation': '#FFD93D'   # Yellow for validation
    }
    
    # Title
    ax.text(8, 11.5, 'PEECOM Framework: Core Methodology Flow', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Step 1: Data Input
    _add_rounded_box(ax, 6.5, 10, 3, 0.8, "CMOHS Dataset\n2,205 samples, 54 features\n3 temporal blocks", 
                     colors['data'], 'black')
    
    # Step 2: Diagnostic Detection
    _add_rounded_box(ax, 1, 8.5, 4.5, 0.8, "DIAGNOSTIC TOOLS\nRandomForest + Others\n→ Detect block leakage", 
                     colors['diagnostic'], 'white')
    
    # Step 3: PEECOM Testbed (parallel to diagnostic)
    _add_rounded_box(ax, 10.5, 8.5, 4.5, 0.8, "PEECOM TESTBED\nApplication Model\n→ Hydraulic classification", 
                     colors['peecom'], 'white')
    
    # Step 4: Results - Pre-Remediation
    _add_rounded_box(ax, 1, 7, 4.5, 0.8, "DIAGNOSTIC RESULTS\n95.8% block prediction\n→ SEVERE LEAKAGE", 
                     colors['diagnostic'], 'white')
    
    _add_rounded_box(ax, 10.5, 7, 4.5, 0.8, "PEECOM RESULTS\nHigh accuracy\n→ Exploiting artifacts", 
                     colors['peecom'], 'white')
    
    # Step 5: BLAST Remediation (central)
    _add_rounded_box(ax, 6, 5.5, 4, 0.8, "BLAST REMEDIATION\nBlock normalization\n→ Remove systematic bias", 
                     colors['remediation'], 'black')
    
    # Step 6: Post-Remediation Validation (both sides)
    _add_rounded_box(ax, 1, 4, 4.5, 0.8, "DIAGNOSTIC VALIDATION\n33.3% block prediction\n→ Leakage eliminated", 
                     colors['validation'], 'black')
    
    _add_rounded_box(ax, 10.5, 4, 4.5, 0.8, "PEECOM VALIDATION\n33.2% classification\n→ Protected from artifacts", 
                     colors['validation'], 'black')
    
    # Step 7: Framework Success
    _add_rounded_box(ax, 5, 2.5, 6, 0.8, "FRAMEWORK SUCCESS\nBoth diagnostic and application models\nachieve chance-level performance", 
                     colors['validation'], 'black')
    
    # Step 8: Universal Impact
    _add_rounded_box(ax, 3, 1, 10, 0.8, "UNIVERSAL FRAMEWORK\nApplicable to any temporal sensor ML application\nMedical, Automotive, IoT, Industrial", 
                     colors['data'], 'black')
    
    # Add arrows showing the flow
    # From data to both diagnostic and PEECOM
    _add_arrow(ax, 7.5, 10, 3.25, 9.3)  # Data → Diagnostic
    _add_arrow(ax, 8.5, 10, 12.75, 9.3)  # Data → PEECOM
    
    # From tools to results
    _add_arrow(ax, 3.25, 8.5, 3.25, 7.8)  # Diagnostic → Results
    _add_arrow(ax, 12.75, 8.5, 12.75, 7.8)  # PEECOM → Results
    
    # From results to remediation
    _add_arrow(ax, 3.25, 7, 6.5, 6.3)  # Diagnostic results → Remediation
    _add_arrow(ax, 12.75, 7, 9.5, 6.3)  # PEECOM results → Remediation
    
    # From remediation to validation
    _add_arrow(ax, 6.5, 5.5, 3.25, 4.8)  # Remediation → Diagnostic validation
    _add_arrow(ax, 9.5, 5.5, 12.75, 4.8)  # Remediation → PEECOM validation
    
    # To framework success
    _add_arrow(ax, 3.25, 4, 6, 3.3)  # Diagnostic validation → Success
    _add_arrow(ax, 12.75, 4, 10, 3.3)  # PEECOM validation → Success
    
    # To universal impact
    _add_arrow(ax, 8, 2.5, 8, 1.8)  # Success → Universal
    
    # Add method labels on arrows
    ax.text(2, 9, 'Block\nPrediction\nTask', ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.text(14, 9, 'Hydraulic\nClassification\nTask', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.text(8, 6, 'Mean + Covariance\nNormalization', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add key insight box
    ax.text(8, 0.2, '🔑 KEY FLOW: Diagnostic tools detect leakage → BLAST removes it → PEECOM protected', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    plt.title("Core Methodology: From Detection to Protection", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('PEECOM_Core_Methodology_Flow.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_technique_connection_diagram():
    """Create diagram showing how techniques connect"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    colors = {
        'input': '#E8E8E8',      # Gray for input
        'method': '#87CEEB',     # Sky blue for methods  
        'model': '#FFB6C1',      # Light pink for models
        'output': '#98FB98'      # Light green for outputs
    }
    
    ax.text(7, 9.5, 'PEECOM Framework: Technique Connections', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input Data
    _add_rounded_box(ax, 5.5, 8.5, 3, 0.6, "Hydraulic Sensor Data\n(temporal blocks)", 
                     colors['input'], 'black')
    
    # Two parallel paths
    # LEFT PATH: Diagnostic
    _add_rounded_box(ax, 1, 7, 3, 0.6, "RandomForest\n(Diagnostic Tool)", colors['method'], 'black')
    _add_rounded_box(ax, 1, 6, 3, 0.6, "Block Prediction Task\n(Detect leakage)", colors['model'], 'black')
    _add_rounded_box(ax, 1, 5, 3, 0.6, "95.8% Accuracy\n(Leakage found)", colors['output'], 'black')
    
    # RIGHT PATH: Application
    _add_rounded_box(ax, 10, 7, 3, 0.6, "PEECOM Model\n(Application Tool)", colors['method'], 'black')
    _add_rounded_box(ax, 10, 6, 3, 0.6, "Hydraulic Classification\n(Normal/Fault)", colors['model'], 'black')
    _add_rounded_box(ax, 10, 5, 3, 0.6, "High Accuracy\n(Also exploiting)", colors['output'], 'black')
    
    # CENTRAL: Remediation
    _add_rounded_box(ax, 5.5, 3.5, 3, 0.6, "BLAST Remediation\n(Block normalization)", colors['method'], 'black')
    
    # Post-remediation results
    _add_rounded_box(ax, 1, 2, 3, 0.6, "33.3% Accuracy\n(Leakage removed)", colors['output'], 'black')
    _add_rounded_box(ax, 10, 2, 3, 0.6, "33.2% Accuracy\n(Model protected)", colors['output'], 'black')
    
    # Success
    _add_rounded_box(ax, 5.5, 0.5, 3, 0.6, "Framework Success\n(Both models clean)", colors['output'], 'black')
    
    # Add connections
    _add_arrow(ax, 6.5, 8.5, 2.5, 7.6)  # Data → Diagnostic
    _add_arrow(ax, 7.5, 8.5, 11.5, 7.6)  # Data → PEECOM
    
    _add_arrow(ax, 2.5, 7, 2.5, 6.6)  # Diagnostic → Task
    _add_arrow(ax, 11.5, 7, 11.5, 6.6)  # PEECOM → Task
    
    _add_arrow(ax, 2.5, 6, 2.5, 5.6)  # Task → Results
    _add_arrow(ax, 11.5, 6, 11.5, 5.6)  # Task → Results
    
    _add_arrow(ax, 2.5, 5, 6, 4.1)  # Results → Remediation
    _add_arrow(ax, 11.5, 5, 8, 4.1)  # Results → Remediation
    
    _add_arrow(ax, 6, 3.5, 2.5, 2.6)  # Remediation → Clean diagnostic
    _add_arrow(ax, 8, 3.5, 11.5, 2.6)  # Remediation → Clean PEECOM
    
    _add_arrow(ax, 2.5, 2, 6, 1.1)  # Clean results → Success
    _add_arrow(ax, 11.5, 2, 8, 1.1)  # Clean results → Success
    
    plt.title("Technique Flow: Detection → Remediation → Protection", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('PEECOM_Technique_Connections.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def _add_rounded_box(ax, x, y, width, height, text, facecolor, textcolor):
    """Add a rounded rectangle with text"""
    box = FancyBboxPatch((x, y), width, height,
                       boxstyle="round,pad=0.05",
                       facecolor=facecolor,
                       edgecolor='black',
                       linewidth=1.5)
    ax.add_patch(box)
    
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color=textcolor,
            wrap=True)

def _add_arrow(ax, x1, y1, x2, y2):
    """Add an arrow between two points"""
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle='->', 
                                  mutation_scale=15,
                                  color='black',
                                  linewidth=1.5)
    ax.add_patch(arrow)

if __name__ == "__main__":
    print("🎨 Generating Simplified PEECOM Core Methodology Flowcharts...")
    
    print("📊 Creating core methodology flow...")
    create_core_methodology_flowchart()
    
    print("🔗 Creating technique connections diagram...")
    create_technique_connection_diagram()
    
    print("\n✅ Simplified flowcharts created successfully!")
    print("Files generated:")
    print("  1. PEECOM_Core_Methodology_Flow.png")
    print("  2. PEECOM_Technique_Connections.png")