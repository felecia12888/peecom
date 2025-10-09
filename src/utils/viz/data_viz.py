#!/usr/bin/env python3
"""
Data Visualizer

Specialized visualizer for dataset analysis, sensor data exploration, and preprocessing insights.
Inherits from BaseVisualizer for consistent styling.
"""

from .base_visualizer import BaseVisualizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataVisualizer(BaseVisualizer):
    """
    Visualizer for dataset analysis and sensor data exploration.

    Creates publication-quality plots for:
    - Sensor data distributions
    - Temporal patterns
    - Feature correlations
    - Preprocessing effects
    - Target class distributions
    """

    def __init__(self, data_dir='dataset/cmohs', **kwargs):
        """
        Initialize data visualizer.

        Args:
            data_dir: Directory containing dataset files
            **kwargs: Arguments passed to BaseVisualizer
        """
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)

    def load_sensor_data(self):
        """Load all sensor data files."""
        sensor_files = {
            'CE': 'Cooling efficiency (%)',
            'CP': 'Cooling power (kW)',
            'EPS1': 'Efficiency factor',
            'FS1': 'Volume flow (l/min)',
            'FS2': 'Volume flow (l/min)',
            'PS1': 'Pressure (bar)',
            'PS2': 'Pressure (bar)',
            'PS3': 'Pressure (bar)',
            'PS4': 'Pressure (bar)',
            'PS5': 'Pressure (bar)',
            'PS6': 'Pressure (bar)',
            'SE': 'Motor power (W)',
            'TS1': 'Temperature (°C)',
            'TS2': 'Temperature (°C)',
            'TS3': 'Temperature (°C)',
            'TS4': 'Temperature (°C)',
            'VS1': 'Vibration (mm/s)'
        }

        sensor_data = {}
        for sensor, description in sensor_files.items():
            file_path = self.data_dir / f"{sensor}.txt"
            if file_path.exists():
                data = pd.read_csv(file_path, sep='\t', header=None)
                sensor_data[sensor] = {
                    'data': data,
                    'description': description
                }

        return sensor_data

    def load_profile_data(self):
        """Load hydraulic system profile data."""
        profile_file = self.data_dir / "profile.txt"
        if profile_file.exists():
            return pd.read_csv(profile_file, sep='\t')
        return None

    def create_sensor_overview(self, figsize=(8, 6)):
        """
        Create individual sensor data overview plots.

        Args:
            figsize: Figure size tuple for each individual plot
        """
        sensor_data = self.load_sensor_data()

        if not sensor_data:
            print("No sensor data found!")
            return None

        saved_plots = {}

        for sensor, info in sensor_data.items():
            # Create individual figure for each sensor
            fig, ax = self.create_figure(figsize=figsize)

            # First column contains the sensor values
            data = info['data'].iloc[:, 0]

            # Create histogram with KDE
            ax.hist(data, bins=50, alpha=0.7, density=True,
                    color=self.get_target_color('cooler_condition'))

            # Add KDE curve
            try:
                data_clean = data.dropna()
                if len(data_clean) > 1:
                    sns.kdeplot(data=data_clean, ax=ax,
                                color='red', linewidth=2)
            except:
                pass

            ax.set_title(
                f'{sensor}: {info["description"]}', fontweight='bold', fontsize=14)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')

            # Add statistics text
            stats_text = f'μ={data.mean():.2f}\nσ={data.std():.2f}\nN={len(data)}'
            ax.text(0.75, 0.75, stats_text, transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10, verticalalignment='top')

            self.add_grid(ax, alpha=0.3)
            self.set_spine_style(ax)

            plt.tight_layout()

            # Save individual plot
            filename = f'{sensor}_sensor_distribution'
            saved_files = self.save_figure(fig, filename)
            saved_plots[sensor] = saved_files

        return saved_plots

    def create_sensor_correlation_matrix(self, figsize=(12, 10)):
        """
        Create correlation matrix heatmap for all sensors.

        Args:
            figsize: Figure size tuple
        """
        sensor_data = self.load_sensor_data()

        if not sensor_data:
            print("No sensor data found!")
            return None

        # Combine all sensor data
        combined_data = {}
        min_length = float('inf')

        for sensor, info in sensor_data.items():
            data = info['data'].iloc[:, 0].values
            combined_data[sensor] = data
            min_length = min(min_length, len(data))

        # Truncate all to same length
        for sensor in combined_data:
            combined_data[sensor] = combined_data[sensor][:min_length]

        df = pd.DataFrame(combined_data)

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Sensor Cross-Correlation Matrix',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def create_temporal_patterns(self, sensors=None, figsize=(10, 6), sample_size=1000):
        """
        Create individual temporal pattern analysis plots for sensors.

        Args:
            sensors: List of sensor names to analyze (None for all sensors)
            figsize: Figure size tuple for each individual plot
            sample_size: Number of samples to plot
        """
        sensor_data = self.load_sensor_data()

        if not sensor_data:
            print("No sensor data found!")
            return None

        # Use all sensors if none specified
        if sensors is None:
            sensors = list(sensor_data.keys())

        saved_plots = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, sensor in enumerate(sensors):
            if sensor not in sensor_data:
                continue

            # Create individual figure for each sensor
            fig, ax = self.create_figure(figsize=figsize)

            data = sensor_data[sensor]['data'].iloc[:sample_size, 0]
            time_index = np.arange(len(data))

            # Plot time series
            ax.plot(time_index, data, color=colors[idx % len(colors)],
                    linewidth=1, alpha=0.8, label=sensor)

            # Add rolling mean
            rolling_mean = data.rolling(window=50, center=True).mean()
            ax.plot(time_index, rolling_mean, color='red', linewidth=2,
                    label=f'{sensor} (50-pt avg)')

            ax.set_xlabel('Time (samples)')
            ax.set_ylabel(f'{sensor_data[sensor]["description"]}')
            ax.set_title(f'{sensor} Temporal Pattern',
                         fontweight='bold', fontsize=14)
            ax.legend(loc='upper right', fontsize=10)

            self.add_grid(ax, alpha=0.3)
            self.set_spine_style(ax)

            plt.tight_layout()

            # Save individual plot
            filename = f'{sensor}_temporal_patterns'
            saved_files = self.save_figure(fig, filename)
            saved_plots[sensor] = saved_files

        return saved_plots

    def create_condition_distribution(self, figsize=(8, 6)):
        """
        Create individual target condition distribution plots.

        Args:
            figsize: Figure size tuple for each individual plot
        """
        profile_data = self.load_profile_data()

        if profile_data is None:
            print("Profile data not found!")
            return None

        # Define target mappings
        targets = {
            # close to total failure, reduced efficiency, full efficiency
            'Cooler condition': [3, 20, 100],
            # optimal switching, small lag, severe lag, close to total failure
            'Valve condition': [100, 90, 80, 73],
            # no leakage, weak leakage, severe leakage
            'Pump leakage': [0, 1, 2],
            # optimal pressure, slightly reduced, severely reduced, close to total failure
            'Accumulator pressure': [130, 115, 100, 90],
            'Stable flag': [0, 1]  # stable, unstable
        }

        saved_plots = {}
        colors = [self.get_target_color('cooler_condition'),
                  self.get_target_color('valve_condition'),
                  self.get_target_color('pump_leakage'),
                  self.get_target_color('accumulator_pressure'),
                  self.get_target_color('stable_flag')]

        for idx, (target, values) in enumerate(targets.items()):
            if target in profile_data.columns:
                # Create individual figure for each target
                fig, ax = self.create_figure(figsize=figsize)

                data = profile_data[target]

                # Create bar plot for categorical distribution
                value_counts = data.value_counts().sort_index()

                bars = ax.bar(range(len(value_counts)), value_counts.values,
                              color=colors[idx % len(colors)], alpha=0.8, edgecolor='black')

                # Add percentage labels
                total = value_counts.sum()
                for bar, count in zip(bars, value_counts.values):
                    height = bar.get_height()
                    percentage = (count / total) * 100
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(value_counts),
                            f'{count}\n({percentage:.1f}%)', ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index)
                ax.set_ylabel('Count')
                ax.set_title(f'{target} Distribution',
                             fontweight='bold', fontsize=14)

                self.add_grid(ax, alpha=0.3)
                self.set_spine_style(ax)

                plt.tight_layout()

                # Save individual plot
                filename = f'{target.lower().replace(" ", "_")}_distribution'
                saved_files = self.save_figure(fig, filename)
                saved_plots[target] = saved_files

        return saved_plots

    def generate_all_data_plots(self):
        """Generate all data visualization plots."""
        print("Generating data visualization plots...")

        plots = {}

        # 1. Sensor overview (individual plots)
        print("Creating individual sensor overview plots...")
        sensor_plots = self.create_sensor_overview()
        if sensor_plots:
            plots.update(sensor_plots)

        # 2. Correlation matrix
        print("Creating correlation matrix...")
        fig2 = self.create_sensor_correlation_matrix()
        if fig2:
            plots['correlation_matrix'] = self.save_figure(
                fig2, 'sensor_correlation_matrix')

        # 3. Temporal patterns (individual plots)
        print("Creating individual temporal pattern plots...")
        temporal_plots = self.create_temporal_patterns()
        if temporal_plots:
            plots.update(temporal_plots)

        # 4. Condition distributions (individual plots)
        print("Creating individual condition distribution plots...")
        condition_plots = self.create_condition_distribution()
        if condition_plots:
            plots.update(condition_plots)

        print(f"Data plots saved to: {self.output_dir}")
        return plots
