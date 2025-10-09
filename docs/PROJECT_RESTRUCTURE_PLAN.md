# PEECOM Project Restructuring Plan

**Date:** October 9, 2025  
**Goal:** Reorganize project with BLAST preprocessing integration and improved model organization

---

## New Directory Structure

```
src/
├── loader/
│   ├── __init__.py
│   ├── config.yaml                    # Unified configuration file
│   ├── dataset_loader.py              # Dataset discovery & registry
│   ├── pipeline_loader.py             # Pipeline orchestration
│   ├── blast_cleaner.py               # BLAST preprocessing (NEW)
│   ├── outlier_remover.py             # Outlier detection/removal (NEW)
│   ├── leakage_filter.py              # Data leakage detection (MOVED)
│   └── peecom_preprocessor.py         # PEECOM-specific preprocessing
│
├── models/
│   ├── peecom/
│   │   ├── __init__.py
│   │   ├── base.py                    # Base PEECOM implementation
│   │   ├── physics_enhanced.py        # PEECOM with advanced physics features
│   │   ├── adaptive.py                # PEECOM with adaptive classifier selection
│   │   └── utils.py                   # PEECOM utilities
│   ├── forest/
│   │   ├── __init__.py
│   │   └── rf.py                      # Random Forest
│   ├── boosting/
│   │   ├── __init__.py
│   │   ├── gbm.py                     # Gradient Boosting
│   │   ├── xgb.py                     # XGBoost (future)
│   │   └── lgbm.py                    # LightGBM (future)
│   ├── linear/
│   │   ├── __init__.py
│   │   └── lr.py                      # Logistic Regression
│   ├── svm/
│   │   ├── __init__.py
│   │   └── svm.py                     # Support Vector Machine
│   ├── nn/
│   │   ├── __init__.py
│   │   └── mlp.py                     # Multi-Layer Perceptron (future)
│   ├── __init__.py
│   └── model_loader.py                # Model registry (UPDATED)
│
├── utils/
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── report.py                  # Report generation
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── model_viz.py               # Model visualizations
│   │   └── performance_viz.py         # Performance plots
│   ├── __init__.py
│   ├── data_utils.py                  # Data utilities (NEW)
│   ├── training_utils.py              # Training utilities (UPDATED)
│   └── results_handler.py             # Results handling
│
├── __init__.py
├── argument_parser.py                 # CLI argument parser
└── main.py                            # Main entry point
```

---

## Key Changes

### 1. **BLAST Integration** (NEW)

- `loader/blast_cleaner.py`: BLAST preprocessing for batch effect removal
- Integrated into preprocessing pipeline
- Supports leave-one-batch-out validation

### 2. **Outlier Removal** (NEW)

- `loader/outlier_remover.py`: Statistical outlier detection and removal
- Multiple methods: IQR, Z-score, Isolation Forest
- Configurable thresholds

### 3. **Leakage Detection** (MOVED)

- Moved from `utils/leakage_detector.py` to `loader/leakage_filter.py`
- Better positioned as preprocessing step
- Enhanced with BLAST-aware detection

### 4. **Model Reorganization**

- Models organized by category (forest, boosting, linear, svm, nn)
- PEECOM variants in dedicated module with descriptive names:
  - `base.py`: Core PEECOM with physics features and hyperparameter optimization
  - `physics_enhanced.py`: PEECOM with advanced feature engineering
  - `adaptive.py`: PEECOM with automatic classifier selection
- Cleaner imports and dependencies

### 5. **Evaluation & Visualization**

- Dedicated `utils/eval/` for metrics and reports
- Dedicated `utils/viz/` for visualizations
- Separation of concerns

---

## Implementation Order

1. ✅ Create directory structure
2. 🔄 Implement BLAST preprocessing (`blast_cleaner.py`)
3. 🔄 Implement outlier removal (`outlier_remover.py`)
4. 🔄 Move & update leakage detection (`leakage_filter.py`)
5. 🔄 Reorganize PEECOM models (`models/peecom/`)
6. 🔄 Reorganize other models (forest, boosting, linear, svm)
7. 🔄 Create evaluation utilities (`utils/eval/`)
8. 🔄 Create visualization utilities (`utils/viz/`)
9. 🔄 Update model loader
10. 🔄 Update main.py and argument parser
11. 🔄 Update configuration files
12. 🔄 Test integration
13. 🧹 Clean up old files

---

## Migration Strategy

### Phase 1: Core Infrastructure (Days 1-2)

- Implement BLAST preprocessing
- Implement outlier removal
- Move leakage detection
- Update pipeline loader

### Phase 2: Model Reorganization (Days 3-4)

- Reorganize PEECOM models
- Reorganize traditional models
- Update model loader
- Test model imports

### Phase 3: Utilities (Day 5)

- Create evaluation utilities
- Create visualization utilities
- Update training utilities

### Phase 4: Integration & Testing (Days 6-7)

- Update main.py
- Update argument parser
- Integration testing
- Performance validation

### Phase 5: Cleanup (Day 8)

- Remove deprecated files
- Update documentation
- Final testing

---

## Files to be Deleted (After Migration)

```
src/config/                             → Moved to loader/config.yaml (unified)
src/utils/leakage_detector.py          → Moved to loader/leakage_filter.py
src/models/simple_peecom.py             → Moved to models/peecom/base.py
src/models/enhanced_peecom.py           → Moved to models/peecom/physics_enhanced.py
src/models/multi_classifier_peecom.py   → Moved to models/peecom/adaptive.py
src/models/random_forest_model.py       → Moved to models/forest/rf.py
src/models/gradient_boosting_model.py   → Moved to models/boosting/gbm.py
src/models/logistic_regression_model.py → Moved to models/linear/lr.py
src/models/svm_model.py                 → Moved to models/svm/svm.py
```

---

## Testing Checklist

- [ ] BLAST preprocessing works correctly
- [ ] Outlier removal works correctly
- [ ] Leakage detection works correctly
- [ ] All models import correctly
- [ ] Training pipeline works end-to-end
- [ ] Cross-validation works
- [ ] Model saving/loading works
- [ ] Visualization generation works
- [ ] CLI arguments work correctly
- [ ] Config files load correctly

---

## Documentation Updates Needed

1. Update README.md with new structure
2. Update USAGE.md with BLAST examples
3. Create BLAST_INTEGRATION.md guide
4. Update API documentation
5. Update examples with new imports

---

**Status:** 🔄 In Progress  
**Next Step:** Implement BLAST preprocessing module
