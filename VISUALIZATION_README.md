# PEECOM Visualization Framework

A comprehensive, publication-quality visualization framework for the PEECOM (Physics-Enhanced Equipment Condition Monitoring) project. This framework generates professional figures suitable for academic papers, presentations, and technical reports.

## Features

- **Publication-Quality Styling**: Consistent, professional styling following academic standards
- **DRY Principles**: Centralized styling, color palettes, and common functionality
- **Multiple Export Formats**: PNG, PDF, and other formats for different use cases
- **Comprehensive Coverage**: Data analysis, model performance, and physics insights
- **Modular Design**: Specialized visualizers for different aspects of the analysis

## Architecture

### Base Components

```
src/visualization/
├── __init__.py                 # Package initialization and exports
├── base_visualizer.py          # Common functionality and styling
├── performance_visualizer.py   # Model performance comparisons
├── data_visualizer.py         # Dataset and sensor analysis
└── model_visualizer.py        # Model-specific insights
```

### Main Scripts

```
generate_visualizations.py     # Main script to generate all figures
test_visualization.py         # Framework testing script
```

## Quick Start

### 1. Install Dependencies

```bash
pip install matplotlib seaborn pandas numpy scikit-learn
```

### 2. Test the Framework

```bash
python test_visualization.py
```

### 3. Generate All Figures

```bash
python generate_visualizations.py
```

### 4. Custom Usage

```python
from src.visualization import PerformanceVisualizer

# Initialize visualizer
viz = PerformanceVisualizer(output_dir='output/figures')

# Generate performance plots
plots = viz.generate_all_performance_plots()
```

## Visualizer Classes

### BaseVisualizer

**Purpose**: Provides common functionality and consistent styling for all visualizers.

**Key Features**:

- Publication-quality matplotlib/seaborn styling
- Consistent color palettes for models and targets
- Common plotting utilities (grids, spines, legends)
- Multi-format figure saving
- Subplot labeling for publications

**Color Palettes**:

- Models: PEECOM (blue), Random Forest (magenta), Logistic Regression (orange), SVM (red), Gradient Boosting (purple)
- Targets: Distinct colors for each hydraulic system component
- Conditions: Green (good), Orange (medium), Red (critical)

### PerformanceVisualizer

**Purpose**: Model performance analysis and comparison visualizations.

**Generated Plots**:

1. **Accuracy Comparison Heatmap**: Model vs target accuracy matrix
2. **Target-Specific Analysis**: Detailed metrics for each target
3. **PEECOM Physics Insight**: Physics enhancement benefits

**Key Methods**:

- `create_accuracy_comparison()`: Overall performance heatmap
- `create_target_specific_comparison()`: Per-target detailed analysis
- `create_peecom_physics_insight()`: PEECOM vs other models

### DataVisualizer

**Purpose**: Dataset analysis and sensor data exploration.

**Generated Plots**:

1. **Sensor Overview**: Distribution analysis for all sensors
2. **Correlation Matrix**: Cross-sensor correlation heatmap
3. **Temporal Patterns**: Time series analysis of key sensors
4. **Condition Distributions**: Target variable distribution analysis

**Key Methods**:

- `create_sensor_overview()`: Comprehensive sensor data analysis
- `create_sensor_correlation_matrix()`: Correlation analysis
- `create_temporal_patterns()`: Time series visualization
- `create_condition_distribution()`: Target analysis

### ModelVisualizer

**Purpose**: Model-specific insights and feature analysis.

**Generated Plots**:

1. **Feature Importance Comparison**: PEECOM vs Random Forest features
2. **PEECOM Physics Analysis**: Detailed physics feature insights
3. **Model Complexity Comparison**: Complexity metrics across models

**Key Methods**:

- `create_feature_importance_comparison()`: Cross-model feature analysis
- `create_peecom_physics_analysis()`: Physics feature deep-dive
- `create_model_complexity_comparison()`: Model complexity metrics

## Configuration Options

### Command Line Arguments

```bash
python generate_visualizations.py [OPTIONS]

Options:
  --output-dir DIR          Output directory (default: output/figures)
  --format FORMAT           Output formats (default: png pdf)
  --data-dir DIR           Dataset directory (default: dataset/dataset)
  --results-dir DIR        Results directory (default: output/results)
  --models-dir DIR         Models directory (default: output/models)
```

### Programmatic Configuration

```python
# Custom styling theme
viz = BaseVisualizer(output_dir='figures', theme='publication')

# Custom color palette
viz.COLOR_PALETTES['custom'] = {'model1': '#123456', 'model2': '#654321'}

# Custom figure size
fig, ax = viz.create_figure(figsize=(10, 6))
```

## Output Structure

```
output/figures/
├── figure_index.md                    # Comprehensive figure catalog
├── model_accuracy_comparison.png      # Overall accuracy comparison
├── model_accuracy_comparison.pdf
├── target_specific_comparison.png     # Target-specific analysis
├── target_specific_comparison.pdf
├── peecom_physics_insight.png         # PEECOM physics benefits
├── peecom_physics_insight.pdf
├── sensor_data_overview.png           # Sensor data distributions
├── sensor_data_overview.pdf
├── sensor_correlation_matrix.png      # Sensor correlations
├── sensor_correlation_matrix.pdf
├── sensor_temporal_patterns.png       # Time series analysis
├── sensor_temporal_patterns.pdf
├── condition_distributions.png        # Target distributions
├── condition_distributions.pdf
├── feature_importance_comparison.png  # Feature importance
├── feature_importance_comparison.pdf
├── peecom_physics_analysis.png        # Physics feature analysis
├── peecom_physics_analysis.pdf
├── model_complexity_comparison.png    # Model complexity
└── model_complexity_comparison.pdf
```

## Styling Standards

### Publication Quality

- High DPI (300) for crisp prints
- Professional color schemes
- Consistent typography (Times New Roman)
- Proper subplot labeling (a), (b), (c)
- Clean grid and spine styling

### Academic Compliance

- Standard figure sizes for journals
- Multiple export formats (PNG for web, PDF for print)
- Comprehensive figure captions via index
- Professional color blindness-friendly palettes

## Usage Examples

### Generate Single Plot Type

```python
from src.visualization import PerformanceVisualizer

viz = PerformanceVisualizer()
fig = viz.create_accuracy_comparison(performance_data)
viz.save_figure(fig, 'custom_accuracy_plot')
```

### Custom Styling

```python
from src.visualization import BaseVisualizer

class CustomVisualizer(BaseVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override styling
        self.STYLE_CONFIG['font.size'] = 14
        self._setup_style()
```

### Batch Processing

```python
from src.visualization import PublicationVisualizer

viz = PublicationVisualizer(output_dir='paper_figures')
all_plots = viz.generate_all_figures()
viz.create_figure_index(all_plots)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed

   ```bash
   pip install matplotlib seaborn pandas numpy scikit-learn
   ```

2. **Missing Data**: Check that dataset and results directories exist

   ```bash
   ls dataset/dataset/
   ls output/results/
   ls output/models/
   ```

3. **Permission Errors**: Ensure write access to output directory
   ```bash
   mkdir -p output/figures
   chmod 755 output/figures
   ```

### Dependencies

- **matplotlib**: Core plotting functionality
- **seaborn**: Statistical visualizations and styling
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Model analysis utilities

## Contributing

### Adding New Visualizations

1. Create new method in appropriate visualizer class
2. Follow existing naming conventions (`create_plot_name()`)
3. Use BaseVisualizer styling utilities
4. Add to `generate_all_*_plots()` method
5. Update figure index descriptions

### Extending Base Functionality

1. Add new utilities to `BaseVisualizer`
2. Maintain backward compatibility
3. Update color palettes in `COLOR_PALETTES`
4. Document new features

## Citation

If you use this visualization framework in your research, please cite:

```bibtex
@software{peecom_visualization,
  title={PEECOM Visualization Framework},
  author={PEECOM Team},
  year={2024},
  url={https://github.com/your-repo/peecom}
}
```
