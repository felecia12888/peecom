#!/usr/bin/env python3
"""
Complete Visualization Suite Runner
==================================

Runs all visualization scripts with A4-optimized outputs:
- Original analysis (for reference)
- A4-optimized versions (for printing/publication)
- Excel tables and data exports
- Comprehensive summary
"""

import subprocess
import sys
from pathlib import Path
import time

class VisualizationSuiteRunner:
    """Run complete visualization suite with A4 optimization"""
    
    def __init__(self):
        self.scripts = [
            ("enhance_performance_metrics.py", "Enhanced Metrics Calculation"),
            ("comprehensive_metrics_dashboard.py", "Comprehensive Metrics Dashboard"),
            ("a4_optimized_visualizer.py", "A4-Optimized Visualizations"),
            ("advanced_model_analysis.py", "Advanced Scientific Analysis")
        ]
        
    def run_script(self, script_name, description):
        """Run a single visualization script"""
        print(f"\n🎨 Running: {description}")
        print("-" * 50)
        
        try:
            result = subprocess.run(
                [sys.executable, script_name], 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                if result.stdout:
                    # Print only key success messages
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if any(keyword in line for keyword in ['✅', '🎉', '📊', '📈', '🏆']):
                            print(f"   {line}")
            else:
                print(f"❌ {description} - FAILED")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} - TIMEOUT")
        except Exception as e:
            print(f"❌ {description} - ERROR: {e}")
    
    def create_summary_report(self):
        """Create final summary of all generated outputs"""
        
        print(f"\n📋 VISUALIZATION SUITE SUMMARY")
        print("=" * 60)
        
        # Check output directories
        output_dirs = [
            "output/figures/a4_optimized",
            "output/figures/comprehensive_metrics", 
            "output/figures/advanced_analysis",
            "output/figures/publication_quality"
        ]
        
        total_files = 0
        for output_dir in output_dirs:
            dir_path = Path(output_dir)
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                total_files += len(files)
                print(f"📁 {output_dir}: {len(files)} files")
        
        print(f"\n📊 Total visualization files generated: {total_files}")
        
        # List key outputs
        key_outputs = [
            "output/figures/a4_optimized/performance_matrix_a4.png",
            "output/figures/a4_optimized/comparison_charts_a4.png", 
            "output/figures/a4_optimized/summary_table_a4.png",
            "output/figures/a4_optimized/comprehensive_performance_metrics.xlsx",
            "output/enhanced_performance_summary.csv",
            "COMPREHENSIVE_ANALYSIS_SUMMARY.md"
        ]
        
        print(f"\n🎯 KEY OUTPUTS FOR A4 PRINTING:")
        for output_file in key_outputs:
            file_path = Path(output_file)
            if file_path.exists():
                print(f"✅ {output_file}")
            else:
                print(f"❌ {output_file} (missing)")
        
        print(f"\n📄 RECOMMENDED FOR A4 PRINTING:")
        print("   📊 performance_matrix_a4.png - Performance comparison heatmaps")
        print("   📈 comparison_charts_a4.png - Detailed analysis charts")
        print("   📋 summary_table_a4.png - Comprehensive metrics table")
        print("   📊 comprehensive_performance_metrics.xlsx - Excel data")
        
        print(f"\n🔍 FONT SIZING OPTIMIZATIONS:")
        print("   ✅ Font sizes reduced to 7-10pt for A4 readability")
        print("   ✅ Compact metric names to prevent overlapping")
        print("   ✅ Optimized spacing and layout")
        print("   ✅ Professional aspect ratios")
        
    def run_complete_suite(self):
        """Run complete visualization suite"""
        
        print("🚀 Complete Visualization Suite Runner")
        print("=" * 60)
        print("📄 A4-optimized outputs with proper font sizing")
        print("📊 Excel tables and comprehensive analysis")
        print("🎨 Professional scientific visualizations")
        
        start_time = time.time()
        
        # Run all visualization scripts
        for script_name, description in self.scripts:
            script_path = Path(script_name)
            if script_path.exists():
                self.run_script(script_name, description)
            else:
                print(f"⚠️  Script not found: {script_name}")
        
        # Create summary
        self.create_summary_report()
        
        # Final timing
        elapsed = time.time() - start_time
        print(f"\n🎉 COMPLETE VISUALIZATION SUITE FINISHED!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds")
        print(f"📁 Check outputs in: output/figures/a4_optimized/")

def main():
    """Main execution function"""
    
    try:
        runner = VisualizationSuiteRunner()
        runner.run_complete_suite()
        
    except Exception as e:
        print(f"❌ Error running visualization suite: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())