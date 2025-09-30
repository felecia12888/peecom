# PEECOM/BLAST Framework - Source Code Organization

This directory contains the organized source code for the PEECOM/BLAST framework.

## Directory Structure

### Core Directories

- **`models/`** - Core PEECOM framework implementations
  - `simple_peecom.py` - Simple PEECOM testbed
  - `enhanced_peecom.py` - Enhanced PEECOM testbed
  - `multi_classifier_peecom.py` - Multi-classifier implementations
  - Various ML model implementations (RF, SVM, LR, etc.)

- **`utils/`** - Utility functions and helpers
  - `training_utils.py` - Training utilities
  - `argument_parser.py` - Command line argument parsing
  - `results_handler.py` - Results processing
  - `time_series_scaler.py` - Data scaling utilities

- **`visualization/`** - All visualization and plotting code
  - Framework schematics and diagrams
  - Performance visualizers
  - Publication-quality plots
  - Model comparison visualizations

- **`analysis/`** - Analysis and validation scripts
  - Statistical validation
  - Performance analysis
  - Metrics computation
  - Novelty validation

- **`experiments/`** - Experimental validation scripts
  - Robustness testing
  - Cross-validation experiments
  - Comparison studies

- **`scripts/`** - Utility and generation scripts
  - Data preprocessing
  - Report generation
  - Enhancement scripts

- **`loader/`** - Data loading utilities
- **`config/`** - Configuration files
- **`documentation/`** - All markdown documentation files

### Root Level Files

- **`run.sh`** - Main execution script
- **`__init__.py`** - Package initialization

## Usage

From the root directory:
- Run main analysis: `python main.py`
- Compare models: `python compare_models.py`
- Execute full pipeline: `bash src/run.sh`

For specific components, navigate to the appropriate subdirectory within `src/`.