# PEECOM - Predictive Equipment Efficiency Condition Monitoring

A comprehensive machine learning framework for hydraulic system condition monitoring and fault detection based on the ZeMA hydraulic test rig dataset. This framework addresses the condition assessment of hydraulic systems using multi-sensor data with four fault types across multiple severity grades.

## 📁 Project Structure

```
peecom/
├── dataset/                    # Raw dataset storage
│   └── cmohs/                 # CMOHS dataset (default)
├── output/                    # All outputs go here
│   ├── processed_data/        # Processed data files
│   ├── analysis/              # Analysis results and reports
│   ├── figures/               # All visualizations and plots
│   ├── logs/                  # Application logs
│   ├── models/                # Trained models
│   ├── metrics/               # Performance metrics
│   ├── predictions/           # Model predictions
│   └── reports/               # Summary reports
├── src/                       # Source code modules
│   ├── config/               # Configuration files
│   ├── loader/               # Data loading and preprocessing
│   ├── models/               # Model definitions and training
│   └── utils/                # Utility functions
├── argument_parser.py        # Command-line argument parser
├── dataset_preprocessing.py  # Data preprocessing entry point
├── main.py                  # Main training entry point
└── README.md               # This file
```

## � Installation & Setup

### Prerequisites

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn pyyaml tqdm
```

### Optional Dependencies

```bash
pip install tensorflow keras-tuner  # For deep learning models
pip install cvxpy                   # For model predictive control
```

### Verify Installation

```bash
python test_workflow.py
```

## �🚀 Quick Start

### 1. Dataset Analysis & Checking

First, analyze your dataset to understand its structure and identify issues:

```bash
# Run dataset analysis and checking
python -c "
from src.loader.dataset_checker import analyze_dataset
results = analyze_dataset('dataset/cmohs', 'output/analysis', 'output/figures')
print('Dataset analysis completed. Check output/analysis/ and output/figures/ directories for results.')
"
```

This will generate analysis results in the `output/` directory including:

- **output/analysis/**: Sensor health reports, correlation analysis, maintenance recommendations
- **output/figures/**: Temporal patterns analysis, correlation plots, condition distributions
- **output/logs/**: Analysis logs and processing information

### 2. Data Preprocessing

Process the raw data based on analysis results:

```bash
# Check

python src/loader/dataset_checker.py dataset/cmohs

# Basic preprocessing
python dataset_preprocessing.py --dataset cmohs

# Advanced preprocessing with custom splits
python dataset_preprocessing.py \
    --dataset cmohs \
    --enforce_split \
    --train_split 0.70 \
    --val_split 0.15 \
    --test_split 0.15 \
    --apply_corrections \
    --remove_outliers

# Preprocessing with specific configuration
python dataset_preprocessing.py \
    --dataset cmohs \
    --config src/config/config.yaml \
    --batch_size 32 \
    --timesteps 10 \
    --features 60
```

**Output**: All processed data will be saved to `output/processed_data/` including:

- Training, validation, and test datasets (X_train.npy, X_val.npy, X_test.npy, etc.)
- Processing metadata and logs in `output/logs/preprocessing.log`
- Processing reports in `output/reports/`

### 3. Model Training

Train models using the processed data:

```bash
# Basic training
python main.py --mode train --dataset cmohs

# Advanced training with hyperparameter tuning
python main.py \
    --mode train \
    --dataset cmohs \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --use_tuning \
    --cross_validation \
    --model_type advanced

# Resume training from checkpoint
python main.py \
    --mode train \
    --resume_from output/checkpoints/best_model.h5 \
    --dataset cmohs
```

**Output**: All training outputs will be saved to `output/` including:

- Trained models in `output/models/`
- Training logs in `output/logs/`
- Checkpoints in `output/checkpoints/`
- Training metrics in `output/metrics/`

### 4. Model Evaluation

Evaluate trained models:

```bash
# Evaluate model
python main.py --mode evaluate --model_path output/models/best_model.h5

# Generate predictions
python main.py --mode predict --model_path output/models/best_model.h5 --input_data output/processed_data/test.npy
```

**Output**: Evaluation results will be saved to:

- Performance metrics in `output/metrics/`
- Predictions in `output/predictions/`
- Evaluation reports in `output/reports/`

## 🔧 Workflow Details

### Phase 1: Dataset Analysis

1. **Run Dataset Checker**: Analyze raw data structure and quality
2. **Review Results**: Check `analysis/dataset_analysis_results.txt` for insights
3. **Update Configuration**: Modify `src/config/config.yaml` based on findings

### Phase 2: Data Preprocessing

1. **Configure Parameters**: Set preprocessing options via CLI arguments
2. **Process Data**: Run `dataset_preprocessing.py` with desired options
3. **Verify Output**: Check `output/processed_data/` directory for processed datasets

### Phase 3: Model Training

1. **Select Configuration**: Choose appropriate model and training parameters
2. **Train Model**: Run `main.py` with training mode
3. **Monitor Progress**: Check logs in `output/logs/` and validation metrics

### Phase 4: Evaluation & Deployment

1. **Evaluate Performance**: Test model on validation/test sets
2. **Generate Reports**: Create performance and analysis reports in `output/reports/`
3. **Deploy Model**: Use trained model from `output/models/` for inference

## 📊 Dataset Information

### ZeMA Hydraulic Test Rig Dataset

- **Instances**: 2,205 hydraulic cycles
- **Attributes**: 43,680 sensor measurements per cycle
- **Duration**: 60-second constant load cycles
- **Components Monitored**: Cooler, Valve, Pump, Hydraulic Accumulator
- **Task Types**: Multi-target classification and regression

### Sensor Configuration

| Sensor  | Physical Quantity            | Unit  | Sampling Rate | Attributes/Cycle |
| ------- | ---------------------------- | ----- | ------------- | ---------------- |
| PS1-PS6 | Pressure                     | bar   | 100 Hz        | 6,000 each       |
| EPS1    | Motor Power                  | W     | 100 Hz        | 6,000            |
| FS1-FS2 | Volume Flow                  | l/min | 10 Hz         | 600 each         |
| TS1-TS4 | Temperature                  | °C    | 1 Hz          | 60 each          |
| VS1     | Vibration                    | mm/s  | 1 Hz          | 60               |
| CE      | Cooling Efficiency (virtual) | %     | 1 Hz          | 60               |
| CP      | Cooling Power (virtual)      | kW    | 1 Hz          | 60               |
| SE      | Efficiency Factor            | %     | 1 Hz          | 60               |

### Target Conditions (profile.txt)

1. **Cooler Condition** (%)

   - 100: Full efficiency (741 instances)
   - 20: Reduced efficiency (732 instances)
   - 3: Close to total failure (732 instances)

2. **Valve Condition** (%)

   - 100: Optimal switching behavior (1,125 instances)
   - 90: Small lag (360 instances)
   - 80: Severe lag (360 instances)
   - 73: Close to total failure (360 instances)

3. **Internal Pump Leakage**

   - 0: No leakage (1,221 instances)
   - 1: Weak leakage (492 instances)
   - 2: Severe leakage (492 instances)

4. **Hydraulic Accumulator** (bar)

   - 130: Optimal pressure (599 instances)
   - 115: Slightly reduced pressure (399 instances)
   - 100: Severely reduced pressure (399 instances)
   - 90: Close to total failure (808 instances)

5. **Stable Flag**
   - 0: Conditions were stable (1,449 instances)
   - 1: Static conditions might not have been reached (756 instances)

## 📊 Available Datasets

- **cmohs**: ZeMA Hydraulic Systems Dataset (default)
- **custom**: Custom dataset (specify path with --dataset_path)

## ⚙️ Configuration

### Config File Structure (`src/config/config.yaml`)

```yaml
data:
  dataset_dir: "dataset/cmohs"
  processed_dir: "processed_data"
  splits:
    train: 0.7
    val: 0.15
    test: 0.15

model:
  input_timesteps: 10
  batch_size: 32
  learning_rate: 0.001
  epochs: 100

preprocessing:
  apply_corrections: true
  remove_outliers: true
  normalize: true
  feature_engineering: true
```

## 🔍 Available Arguments

### Dataset Preprocessing Arguments

- `--dataset`: Dataset name (default: cmohs)
- `--enforce_split`: Force data splitting
- `--train_split`: Training split ratio (default: 0.7)
- `--val_split`: Validation split ratio (default: 0.15)
- `--test_split`: Test split ratio (default: 0.15)
- `--output_dir`: Output directory for processed data (default: output/processed_data)
- `--apply_corrections`: Apply sensor corrections
- `--remove_outliers`: Remove outlier data points

### Training Arguments

- `--mode`: Operation mode (train/evaluate/predict)
- `--dataset`: Dataset to use
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate
- `--model_type`: Model architecture type
- `--use_tuning`: Enable hyperparameter tuning
- `--cross_validation`: Use cross-validation

## 📈 Output Files

All outputs are organized in the `output/` directory:

### Preprocessing Outputs

- `output/processed_data/X_train.npy`: Training features
- `output/processed_data/X_val.npy`: Validation features
- `output/processed_data/X_test.npy`: Test features
- `output/processed_data/y_train.npy`: Training labels
- `output/processed_data/y_val.npy`: Validation labels
- `output/processed_data/y_test.npy`: Test labels
- `output/processed_data/metadata.json`: Processing metadata
- `output/logs/preprocessing.log`: Preprocessing logs

### Analysis Outputs

- `output/analysis/dataset_analysis_results.txt`: Comprehensive analysis report
- `output/analysis/dataset_analysis_results.csv`: Analysis data in CSV format
- `output/figures/*.png`: All visualization plots (temporal patterns, correlations, etc.)

### Training Outputs

- `output/models/`: Trained model files (.h5, .pkl)
- `output/logs/`: Training logs and metrics
- `output/checkpoints/`: Model checkpoints during training
- `output/metrics/`: Training and validation metrics
- `output/reports/`: Training summary reports

### Evaluation Outputs

- `output/predictions/`: Model predictions on test data
- `output/metrics/`: Evaluation metrics and performance scores
- `output/reports/`: Evaluation summary reports

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Not Found**: Ensure dataset exists in `dataset/` directory
2. **Memory Issues**: Reduce batch size or use data generators
3. **Performance Issues**: Check data quality and model parameters

### Debug Mode

Enable verbose logging:

```bash
python main.py --mode train --verbose --debug
```

## 📚 API Reference

### Core Classes

- `PEECOMDataLoader`: Data loading and basic preprocessing
- `PEECOMDataProcessor`: Advanced data processing and feature engineering
- `SensorValidator`: Sensor data validation and quality checks
- `SensorMonitor`: Real-time sensor monitoring

### Key Functions

- `analyze_dataset()`: Comprehensive dataset analysis
- `preprocess_data()`: Data preprocessing pipeline
- `train_model()`: Model training pipeline
- `evaluate_model()`: Model evaluation pipeline

## 🤝 Contributing

1. Follow the established code structure
2. Add tests for new functionality
3. Update documentation for new features
4. Use type hints and docstrings

## 📄 License

[Add your license information here]
