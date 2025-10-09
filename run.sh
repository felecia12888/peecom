#!/usr/bin/env bash
################################################################################
# PEECOM Experiment Runner
# 
# Robust script for running PEECOM experiments with different models and datasets
# Supports both simple and complex argument passing
#
# Usage:
#   ./run.sh                          # Run default configuration
#   ./run.sh --dataset cmohs          # Run on specific dataset
#   ./run.sh --model peecom_base      # Run with specific model
#   ./run.sh --all-models             # Run all model types
#   ./run.sh --all-datasets           # Run all available datasets
#   ./run.sh --help                   # Show help message
################################################################################

set -e  # Exit on error
# set -x  # Uncomment for debugging

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Default configuration
DEFAULT_DATASET="cmohs"
DEFAULT_MODEL="peecom"
DEFAULT_TARGET="stable_flag"
PYTHON_CMD="python3"

# Available models in new structure
PEECOM_MODELS=("peecom" "peecom_base" "peecom_physics" "peecom_adaptive")
TRADITIONAL_MODELS=("random_forest" "gradient_boosting" "logistic_regression" "svm")
ALL_MODELS=("${PEECOM_MODELS[@]}" "${TRADITIONAL_MODELS[@]}")

# Available datasets (auto-detect from processed_data/)
AVAILABLE_DATASETS=()
if [ -d "output/processed_data" ]; then
    for dir in output/processed_data/*/; do
        if [ -d "$dir" ]; then
            dataset_name=$(basename "$dir")
            AVAILABLE_DATASETS+=("$dataset_name")
        fi
    done
fi

# Fallback datasets if no processed data found
if [ ${#AVAILABLE_DATASETS[@]} -eq 0 ]; then
    AVAILABLE_DATASETS=("cmohs" "equipmentad" "mlclassem" "motorvd" "multivariatetsd" "sensord" "smartmd")
fi

################################################################################
# Helper Functions
################################################################################

show_help() {
    cat << EOF
PEECOM Experiment Runner

Usage: $0 [OPTIONS]

Options:
    --dataset <name>        Specify dataset (default: $DEFAULT_DATASET)
    --model <name>          Specify model (default: $DEFAULT_MODEL)
    --target <name>         Specify target variable (default: $DEFAULT_TARGET)
    --eval-all              Evaluate all targets in dataset
    --all-models            Run all available models
    --all-datasets          Run all available datasets
    --peecom-only           Run only PEECOM variants
    --traditional-only      Run only traditional ML models
    --visualize             Generate visualizations after training
    --use-blast             Enable BLAST preprocessing
    --remove-outliers       Enable outlier removal
    --check-leakage         Enable data leakage detection
    --list-models           List all available models
    --list-datasets         List all available datasets
    --help                  Show this help message

Available Models:
    PEECOM Variants:
$(printf '        - %s\n' "${PEECOM_MODELS[@]}")
    
    Traditional ML:
$(printf '        - %s\n' "${TRADITIONAL_MODELS[@]}")

Available Datasets:
$(printf '    - %s\n' "${AVAILABLE_DATASETS[@]}")

Examples:
    # Train PEECOM on default dataset
    $0 --model peecom

    # Train specific model on specific dataset
    $0 --dataset cmohs --model random_forest --target valve_condition

    # Evaluate all targets
    $0 --dataset cmohs --model peecom_physics --eval-all

    # Run all PEECOM variants on a dataset
    $0 --dataset cmohs --peecom-only --eval-all

    # Run all models with BLAST preprocessing
    $0 --dataset cmohs --all-models --use-blast --eval-all

    # Run comprehensive comparison
    $0 --all-datasets --all-models --eval-all --visualize

EOF
}

log_info() {
    echo "[INFO] $*"
}

log_success() {
    echo "[✓] $*"
}

log_error() {
    echo "[✗] $*" >&2
}

run_experiment() {
    local dataset="$1"
    local model="$2"
    local additional_args=("${@:3}")
    
    log_info "Running: Dataset=$dataset, Model=$model"
    log_info "Command: $PYTHON_CMD main.py --dataset \"$dataset\" --model \"$model\" ${additional_args[*]}"
    
    if $PYTHON_CMD main.py --dataset "$dataset" --model "$model" "${additional_args[@]}"; then
        log_success "Completed: $model on $dataset"
        return 0
    else
        log_error "Failed: $model on $dataset"
        return 1
    fi
}

################################################################################
# Parse Arguments
################################################################################

DATASET="$DEFAULT_DATASET"
MODEL="$DEFAULT_MODEL"
TARGET="$DEFAULT_TARGET"
EVAL_ALL=false
ALL_MODELS_FLAG=false
ALL_DATASETS_FLAG=false
PEECOM_ONLY=false
TRADITIONAL_ONLY=false
VISUALIZE=false
USE_BLAST=false
REMOVE_OUTLIERS=false
CHECK_LEAKAGE=false
ADDITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            ADDITIONAL_ARGS+=("--target" "$2")
            shift 2
            ;;
        --eval-all)
            EVAL_ALL=true
            ADDITIONAL_ARGS+=("--eval-all")
            shift
            ;;
        --all-models)
            ALL_MODELS_FLAG=true
            shift
            ;;
        --all-datasets)
            ALL_DATASETS_FLAG=true
            shift
            ;;
        --peecom-only)
            PEECOM_ONLY=true
            shift
            ;;
        --traditional-only)
            TRADITIONAL_ONLY=true
            shift
            ;;
        --visualize)
            VISUALIZE=true
            ADDITIONAL_ARGS+=("--visualize")
            shift
            ;;
        --use-blast)
            USE_BLAST=true
            ADDITIONAL_ARGS+=("--use-blast")
            shift
            ;;
        --remove-outliers)
            REMOVE_OUTLIERS=true
            ADDITIONAL_ARGS+=("--remove-outliers")
            shift
            ;;
        --check-leakage)
            CHECK_LEAKAGE=true
            ADDITIONAL_ARGS+=("--check-leakage")
            shift
            ;;
        --list-models)
            $PYTHON_CMD main.py --list-models
            exit 0
            ;;
        --list-datasets)
            $PYTHON_CMD main.py --list-datasets
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

################################################################################
# Main Execution
################################################################################

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        PEECOM EXPERIMENT RUNNER                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Determine which models to run
MODELS_TO_RUN=()
if [ "$PEECOM_ONLY" = true ]; then
    MODELS_TO_RUN=("${PEECOM_MODELS[@]}")
    log_info "Running PEECOM variants only"
elif [ "$TRADITIONAL_ONLY" = true ]; then
    MODELS_TO_RUN=("${TRADITIONAL_MODELS[@]}")
    log_info "Running traditional ML models only"
elif [ "$ALL_MODELS_FLAG" = true ]; then
    MODELS_TO_RUN=("${ALL_MODELS[@]}")
    log_info "Running all models"
else
    MODELS_TO_RUN=("$MODEL")
fi

# Determine which datasets to run
DATASETS_TO_RUN=()
if [ "$ALL_DATASETS_FLAG" = true ]; then
    DATASETS_TO_RUN=("${AVAILABLE_DATASETS[@]}")
    log_info "Running on all datasets"
else
    DATASETS_TO_RUN=("$DATASET")
fi

# Print configuration
echo "Configuration:"
echo "  Datasets: ${DATASETS_TO_RUN[*]}"
echo "  Models: ${MODELS_TO_RUN[*]}"
echo "  Evaluate All Targets: $EVAL_ALL"
echo "  BLAST Preprocessing: $USE_BLAST"
echo "  Outlier Removal: $REMOVE_OUTLIERS"
echo "  Leakage Detection: $CHECK_LEAKAGE"
echo "  Visualizations: $VISUALIZE"
echo ""

# Run experiments
TOTAL_EXPERIMENTS=$((${#DATASETS_TO_RUN[@]} * ${#MODELS_TO_RUN[@]}))
CURRENT_EXPERIMENT=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

log_info "Starting $TOTAL_EXPERIMENTS experiments..."
echo ""

for dataset in "${DATASETS_TO_RUN[@]}"; do
    for model in "${MODELS_TO_RUN[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        
        echo "════════════════════════════════════════════════════════════"
        echo "Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
        echo "════════════════════════════════════════════════════════════"
        
        if run_experiment "$dataset" "$model" "${ADDITIONAL_ARGS[@]}"; then
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        else
            FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        fi
        
        echo ""
    done
done

# Final summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 EXPERIMENT SUMMARY                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Total Experiments: $TOTAL_EXPERIMENTS"
log_success "Successful: $SUCCESSFUL_EXPERIMENTS"
if [ $FAILED_EXPERIMENTS -gt 0 ]; then
    log_error "Failed: $FAILED_EXPERIMENTS"
fi
echo ""

if [ $FAILED_EXPERIMENTS -eq 0 ]; then
    log_success "All experiments completed successfully! 🎉"
    exit 0
else
    log_error "Some experiments failed. Check logs above for details."
    exit 1
fi
