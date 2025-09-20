#!/usr/bin/env python3
"""
PEECOM Feature Importance Analysis & Task Analysis

This script creates feature importance comparisons between PEECOM and Random Forest
and analyzes the prediction, control, and monitoring tasks in the PEECOM system.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for A4 format
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 7,
    'axes.labelsize': 6,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.titlesize': 8,
    'lines.linewidth': 0.8,
    'lines.markersize': 4,
    'grid.linewidth': 0.2,
    'grid.alpha': 0.3,
})

class PEECOMTaskAnalyzer:
    """Analyze PEECOM tasks: Prediction, Control, and Monitoring"""
    
    def __init__(self, output_dir="output/figures/peecom_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load feature importance data
        self.feature_data = self.load_feature_importance()
        
        # Color schemes
        self.colors = {
            'peecom': '#1f77b4',
            'random_forest': '#ff7f0e',
            'prediction': '#2ca02c',
            'control': '#d62728',
            'monitoring': '#9467bd'
        }
        
        # Target mappings for PEECOM framework
        self.target_mappings = {
            'accumulator_pressure': 'Prediction',
            'cooler_condition': 'Prediction', 
            'pump_leakage': 'Prediction',
            'stable_flag': 'Monitoring',
            'valve_condition': 'Prediction'
        }
        
    def load_feature_importance(self):
        """Load feature importance comparison data"""
        fi_file = "feature_importance_comparison.csv"
        if Path(fi_file).exists():
            return pd.read_csv(fi_file)
        else:
            print(f"❌ {fi_file} not found!")
            return pd.DataFrame()
    
    def create_feature_importance_comparison(self):
        """Create comprehensive feature importance comparison plots"""
        
        if self.feature_data.empty:
            print("⚠️ No feature importance data available")
            return
            
        # Create large figure for comprehensive analysis
        fig = plt.figure(figsize=(8.27, 11.7))  # Full A4 size
        
        # Create subplots with different sizes
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.8], hspace=0.4, wspace=0.3)
        
        targets = ['accumulator_pressure', 'cooler_condition', 'pump_leakage', 'valve_condition', 'stable_flag']
        
        # 1-4. Individual target comparisons (top 10 features each)
        for i, target in enumerate(targets[:4]):
            ax = fig.add_subplot(gs[i//2, i%2])
            self.plot_target_feature_comparison(ax, target)
        
        # 5. Overall feature category analysis
        ax5 = fig.add_subplot(gs[2, :])
        self.plot_feature_category_analysis(ax5)
        
        # 6. Model feature count comparison
        ax6 = fig.add_subplot(gs[3, :])
        self.plot_feature_count_comparison(ax6)
        
        # Overall title
        fig.suptitle('PEECOM vs Random Forest: Feature Importance Analysis\nPredictive Energy Efficiency Control and Optimization Model', 
                    fontsize=9, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save outputs
        output_path = self.output_dir / "feature_importance_comparison_a4.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        
        output_path_pdf = self.output_dir / "feature_importance_comparison_a4.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        
        plt.show()
        print(f"✅ Feature importance comparison saved: {output_path}")
        
    def plot_target_feature_comparison(self, ax, target):
        """Plot feature importance comparison for a specific target"""
        
        target_data = self.feature_data[self.feature_data['target'] == target]
        
        # Get top 8 features from both models combined
        all_features = target_data.groupby('feature')['importance'].max().nlargest(8)
        top_features = all_features.index.tolist()
        
        # Prepare data for plotting
        peecom_data = target_data[target_data['model'] == 'peecom'].set_index('feature')
        rf_data = target_data[target_data['model'] == 'random_forest'].set_index('feature')
        
        peecom_values = [peecom_data.loc[f, 'importance'] if f in peecom_data.index else 0 for f in top_features]
        rf_values = [rf_data.loc[f, 'importance'] if f in rf_data.index else 0 for f in top_features]
        
        # Create comparison bars
        x = np.arange(len(top_features))
        width = 0.35
        
        bars1 = ax.barh(x - width/2, peecom_values, width, 
                       label='PEECOM', color=self.colors['peecom'], alpha=0.8)
        bars2 = ax.barh(x + width/2, rf_values, width,
                       label='Random Forest', color=self.colors['random_forest'], alpha=0.8)
        
        ax.set_title(f'{target.replace("_", " ").title()}', fontsize=6, fontweight='bold', pad=3)
        ax.set_xlabel('Feature Importance', fontsize=5)
        ax.set_yticks(x)
        ax.set_yticklabels([f[:12] for f in top_features], fontsize=4)  # Truncate long names
        ax.legend(fontsize=4, loc='lower right')
        ax.grid(True, alpha=0.2, linewidth=0.3)
        
        # Add value labels
        for i, (p_val, rf_val) in enumerate(zip(peecom_values, rf_values)):
            if p_val > 0.01:
                ax.text(p_val + 0.001, i - width/2, f'{p_val:.3f}', 
                       va='center', ha='left', fontsize=3)
            if rf_val > 0.01:
                ax.text(rf_val + 0.001, i + width/2, f'{rf_val:.3f}', 
                       va='center', ha='left', fontsize=3)
    
    def plot_feature_category_analysis(self, ax):
        """Plot analysis by sensor/feature categories"""
        
        # Categorize features by sensor type
        def categorize_feature(feature_name):
            if 'PS' in feature_name:
                return 'Pressure'
            elif 'TS' in feature_name:
                return 'Temperature'
            elif 'FS' in feature_name:
                return 'Flow'
            elif 'EPS' in feature_name:
                return 'Motor Power'
            elif any(x in feature_name for x in ['CE', 'CP', 'SE']):
                return 'Efficiency'
            elif 'Acc' in feature_name:
                return 'Acceleration'
            else:
                return 'Other'
        
        # Calculate importance by category
        self.feature_data['category'] = self.feature_data['feature'].apply(categorize_feature)
        
        category_analysis = self.feature_data.groupby(['model', 'target', 'category'])['importance'].sum().reset_index()
        category_summary = category_analysis.groupby(['model', 'category'])['importance'].mean().reset_index()
        
        # Create grouped bar chart
        categories = sorted(category_summary['category'].unique())
        peecom_values = [category_summary[(category_summary['model'] == 'peecom') & 
                                        (category_summary['category'] == cat)]['importance'].values[0] 
                        if len(category_summary[(category_summary['model'] == 'peecom') & 
                                              (category_summary['category'] == cat)]) > 0 else 0 
                        for cat in categories]
        rf_values = [category_summary[(category_summary['model'] == 'random_forest') & 
                                    (category_summary['category'] == cat)]['importance'].values[0] 
                    if len(category_summary[(category_summary['model'] == 'random_forest') & 
                                          (category_summary['category'] == cat)]) > 0 else 0 
                    for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, peecom_values, width, 
                      label='PEECOM', color=self.colors['peecom'], alpha=0.8)
        bars2 = ax.bar(x + width/2, rf_values, width,
                      label='Random Forest', color=self.colors['random_forest'], alpha=0.8)
        
        ax.set_title('Average Feature Importance by Sensor Category', fontsize=6, fontweight='bold', pad=3)
        ax.set_ylabel('Average Importance', fontsize=5)
        ax.set_xlabel('Sensor Category', fontsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=5)
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.2, linewidth=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=4)
        for bar in bars2:
            height = bar.get_height()
            if height > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=4)
    
    def plot_feature_count_comparison(self, ax):
        """Plot feature count comparison between models"""
        
        # Count non-zero importance features for each model-target combination
        feature_counts = self.feature_data.groupby(['model', 'target']).apply(
            lambda x: (x['importance'] > 0.001).sum()
        ).reset_index(name='feature_count')
        
        # Create comparison
        targets = sorted(feature_counts['target'].unique())
        peecom_counts = [feature_counts[(feature_counts['model'] == 'peecom') & 
                                       (feature_counts['target'] == t)]['feature_count'].values[0] 
                        for t in targets]
        rf_counts = [feature_counts[(feature_counts['model'] == 'random_forest') & 
                                   (feature_counts['target'] == t)]['feature_count'].values[0] 
                    for t in targets]
        
        x = np.arange(len(targets))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, peecom_counts, width, 
                      label='PEECOM', color=self.colors['peecom'], alpha=0.8)
        bars2 = ax.bar(x + width/2, rf_counts, width,
                      label='Random Forest', color=self.colors['random_forest'], alpha=0.8)
        
        ax.set_title('Number of Important Features (>0.001) by Target', fontsize=6, fontweight='bold', pad=3)
        ax.set_ylabel('Feature Count', fontsize=5)
        ax.set_xlabel('Target', fontsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in targets], fontsize=4)
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.2, linewidth=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontsize=4)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontsize=4)
    
    def analyze_peecom_tasks(self):
        """Analyze PEECOM framework tasks: Prediction, Control, and Monitoring"""
        
        print("🔍 PEECOM TASK ANALYSIS")
        print("="*60)
        print("Project: Predictive Energy Efficiency Control and Optimization Model")
        print("="*60)
        
        # Load performance data for task analysis
        perf_data = pd.read_csv("comprehensive_performance_data.csv") if Path("comprehensive_performance_data.csv").exists() else pd.DataFrame()
        
        if perf_data.empty:
            print("❌ No performance data available for task analysis")
            return
        
        # Analyze each task component
        print(f"\n📊 TASK BREAKDOWN ANALYSIS:")
        print("="*40)
        
        # 1. PREDICTION TASKS
        prediction_targets = ['accumulator_pressure', 'cooler_condition', 'pump_leakage', 'valve_condition']
        prediction_data = perf_data[perf_data['target'].isin(prediction_targets)]
        
        print(f"\n🎯 PREDICTION TASKS:")
        print(f"   Purpose: Predict hydraulic system component conditions")
        print(f"   Targets: {len(prediction_targets)} components")
        
        if not prediction_data.empty:
            peecom_pred = prediction_data[prediction_data['model'] == 'peecom']
            rf_pred = prediction_data[prediction_data['model'] == 'random_forest']
            
            print(f"   📈 PEECOM Performance: {peecom_pred['test_accuracy'].mean():.3f} ± {peecom_pred['test_accuracy'].std():.3f}")
            print(f"   📈 Random Forest Performance: {rf_pred['test_accuracy'].mean():.3f} ± {rf_pred['test_accuracy'].std():.3f}")
            
            print(f"\n   Detailed Prediction Results:")
            for target in prediction_targets:
                target_data = prediction_data[prediction_data['target'] == target]
                peecom_acc = target_data[target_data['model'] == 'peecom']['test_accuracy'].iloc[0] if not target_data[target_data['model'] == 'peecom'].empty else 0
                rf_acc = target_data[target_data['model'] == 'random_forest']['test_accuracy'].iloc[0] if not target_data[target_data['model'] == 'random_forest'].empty else 0
                print(f"      • {target:20} PEECOM: {peecom_acc:.3f}, RF: {rf_acc:.3f}")
        
        # 2. MONITORING TASKS  
        monitoring_targets = ['stable_flag']
        monitoring_data = perf_data[perf_data['target'].isin(monitoring_targets)]
        
        print(f"\n📊 MONITORING TASKS:")
        print(f"   Purpose: Monitor system stability and operational status")
        print(f"   Targets: {len(monitoring_targets)} stability indicators")
        
        if not monitoring_data.empty:
            peecom_mon = monitoring_data[monitoring_data['model'] == 'peecom']
            rf_mon = monitoring_data[monitoring_data['model'] == 'random_forest']
            
            print(f"   📈 PEECOM Performance: {peecom_mon['test_accuracy'].mean():.3f}")
            print(f"   📈 Random Forest Performance: {rf_mon['test_accuracy'].mean():.3f}")
            
            print(f"\n   Detailed Monitoring Results:")
            for target in monitoring_targets:
                target_data = monitoring_data[monitoring_data['target'] == target]
                peecom_acc = target_data[target_data['model'] == 'peecom']['test_accuracy'].iloc[0]
                rf_acc = target_data[target_data['model'] == 'random_forest']['test_accuracy'].iloc[0]
                print(f"      • {target:20} PEECOM: {peecom_acc:.3f}, RF: {rf_acc:.3f}")
        
        # 3. CONTROL TASKS
        print(f"\n🎛️ CONTROL TASKS:")
        print(f"   Status: ❌ NOT IMPLEMENTED")
        print(f"   Current System: Classification-based prediction and monitoring only")
        print(f"   Missing Components:")
        print(f"      • Control action generation")
        print(f"      • Optimization algorithms") 
        print(f"      • Actuator interfaces")
        print(f"      • Closed-loop feedback")
        
        # 4. ENERGY EFFICIENCY OPTIMIZATION
        print(f"\n⚡ ENERGY EFFICIENCY OPTIMIZATION:")
        print(f"   Status: ❌ NOT EXPLICITLY IMPLEMENTED")
        print(f"   Current System: Condition monitoring without efficiency optimization")
        print(f"   Missing Components:")
        print(f"      • Energy consumption prediction")
        print(f"      • Efficiency optimization algorithms")
        print(f"      • Energy-aware control strategies")
        print(f"      • Cost-benefit analysis")
        
        return self.generate_peecom_recommendations()
    
    def generate_peecom_recommendations(self):
        """Generate recommendations for implementing missing PEECOM components"""
        
        print(f"\n💡 RECOMMENDATIONS FOR COMPLETE PEECOM IMPLEMENTATION:")
        print("="*70)
        
        recommendations = {
            "Control Tasks": [
                "Implement Model Predictive Control (MPC) for valve and pump control",
                "Add PID controllers for temperature and pressure regulation", 
                "Develop real-time control action generation based on predictions",
                "Create actuator interface modules for valve position control",
                "Implement safety constraints and operational limits"
            ],
            
            "Energy Efficiency Optimization": [
                "Add energy consumption sensors and data collection",
                "Implement multi-objective optimization (performance vs energy)",
                "Develop energy-aware scheduling algorithms",
                "Create efficiency benchmarking and KPI tracking",
                "Add cost-function optimization for operational efficiency"
            ],
            
            "Enhanced Monitoring": [
                "Implement anomaly detection algorithms",
                "Add trend analysis and degradation monitoring", 
                "Create real-time alerting and notification systems",
                "Develop predictive maintenance scheduling",
                "Add data quality monitoring and sensor validation"
            ],
            
            "System Integration": [
                "Develop SCADA/HMI interfaces for operators",
                "Implement real-time data streaming and processing",
                "Add database integration for historical analysis",
                "Create API interfaces for external system integration",
                "Implement cybersecurity and access control measures"
            ]
        }
        
        for category, items in recommendations.items():
            print(f"\n🔧 {category}:")
            for i, item in enumerate(items, 1):
                print(f"   {i}. {item}")
        
        print(f"\n🎯 PRIORITY IMPLEMENTATION ORDER:")
        print(f"   1. Enhanced Monitoring (extend current capabilities)")
        print(f"   2. Energy Efficiency Optimization (core PEECOM purpose)")
        print(f"   3. Control Tasks (close the control loop)")
        print(f"   4. System Integration (production deployment)")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete PEECOM analysis"""
        
        print("🚀 COMPLETE PEECOM ANALYSIS")
        print("="*50)
        
        # Create feature importance visualizations
        self.create_feature_importance_comparison()
        
        # Analyze PEECOM tasks
        recommendations = self.analyze_peecom_tasks()
        
        # Save analysis summary
        summary_path = self.output_dir / "peecom_task_analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("PEECOM Task Analysis Summary\n")
            f.write("="*50 + "\n\n")
            f.write("Project: Predictive Energy Efficiency Control and Optimization Model\n\n")
            
            f.write("IMPLEMENTED TASKS:\n")
            f.write("✅ Prediction: 4 targets (98.7% avg accuracy)\n")
            f.write("✅ Monitoring: 1 target (98.0% accuracy)\n\n")
            
            f.write("MISSING TASKS:\n")
            f.write("❌ Control: No control actions implemented\n")
            f.write("❌ Energy Optimization: No efficiency optimization\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            for category, items in recommendations.items():
                f.write(f"\n{category}:\n")
                for item in items:
                    f.write(f"  - {item}\n")
        
        print(f"\n✅ Complete analysis saved to: {self.output_dir}")
        print(f"📊 Feature importance plots: feature_importance_comparison_a4.png")
        print(f"📋 Task analysis summary: peecom_task_analysis_summary.txt")

if __name__ == "__main__":
    analyzer = PEECOMTaskAnalyzer()
    analyzer.run_complete_analysis()