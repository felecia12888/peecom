#!/usr/bin/env python3
"""
Quick Model Visualization Launcher
=================================

Simple script to generate all model comparison visualizations with one command.

Usage:
    python create_model_plots.py              # Generate all plots (PNG)
    python create_model_plots.py --pdf        # Generate all plots (PDF)  
    python create_model_plots.py --dataset cmohs  # Only CMOHS dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_visualizations(save_format='png', dataset=None, output_dir=None):
    """Run the model comparison visualizations."""
    
    print("🎨 PEECOM Model Visualization Generator")
    print("=" * 50)
    
    # Check if the main visualization script exists
    viz_script = Path('visualize_model_comparison.py')
    if not viz_script.exists():
        print(f"❌ Error: {viz_script} not found!")
        print("Make sure you're running this from the project root directory.")
        return False
    
    # Build the command
    cmd = [sys.executable, str(viz_script)]
    
    if dataset:
        cmd.extend(['--dataset', dataset])
    
    if save_format:
        cmd.extend(['--save-format', save_format])
        
    if output_dir:
        cmd.extend(['--output-dir', output_dir])
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print()
    
    try:
        # Run the visualization script
        result = subprocess.run(cmd, check=True, capture_output=False)
        print()
        print("✅ Successfully generated all model comparison plots!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running visualization script: {e}")
        return False
    except KeyboardInterrupt:
        print("\\n⚠️  Visualization interrupted by user")
        return False

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Quick launcher for model visualization plots',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pdf', action='store_true',
                       help='Save plots as PDF instead of PNG')
    parser.add_argument('--svg', action='store_true', 
                       help='Save plots as SVG instead of PNG')
    parser.add_argument('--dataset', type=str,
                       help='Generate plots for specific dataset only')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory')
    
    args = parser.parse_args()
    
    # Determine save format
    save_format = 'png'  # default
    if args.pdf:
        save_format = 'pdf'
    elif args.svg:
        save_format = 'svg'
    
    # Run the visualizations
    success = run_visualizations(
        save_format=save_format,
        dataset=args.dataset,
        output_dir=args.output_dir
    )
    
    if success:
        print()
        print("📊 Model comparison plots are ready!")
        output_dir = args.output_dir or 'output/figures/model_comparison'
        print(f"📁 Check the output directory: {output_dir}")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()