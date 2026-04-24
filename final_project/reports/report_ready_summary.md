## Project Status

The project is now in a report-ready state for the holdout baseline phase.
The full training, comparison, validation-analysis, and submission pipelines are implemented and reproducible through `run.py`.

## Final Recommended Configuration

- Model: `resnet18`
- Augmentation: `spectral_safe`
- Input size: `64 x 64`
- Input mode: `grayscale`
- Device used for stable experiments: `cpu`

## Main Experimental Findings

### 1. Stronger baseline selection

Holdout comparison with `aug=none` shows that `resnet18` clearly outperforms the custom `cnn`.

| Model | Accuracy | Macro F1 | Best Epoch |
| --- | ---: | ---: | ---: |
| `resnet18` | 0.9686 | 0.9652 | 18 |
| `cnn` | 0.9177 | 0.9120 | 8 |

Source artifact:
- `experiments/comparisons/baseline_comparison_none.csv`

### 2. Augmentation selection

For `resnet18`, the augmentation ranking is:

| Augmentation | Accuracy | Macro F1 | Rank |
| --- | ---: | ---: | ---: |
| `spectral_safe` | 0.9762 | 0.9715 | 1 |
| `none` | 0.9686 | 0.9652 | 2 |
| `light` | 0.9686 | 0.9644 | 3 |

Source artifact:
- `experiments/comparisons/resnet18_aug_comparison.csv`

## Best Holdout Validation Analysis

Best run:
- `model=resnet18`
- `aug=spectral_safe`

Validation summary:
- Accuracy: `0.9762`
- Macro F1: `0.9715`
- Validation samples: `1846`
- Errors: `44`
- Mean prediction confidence: `0.9790`
- Mean confidence on errors: `0.7863`
- Most common confusion: `1 -> 2` with `8` cases

Source artifacts:
- `experiments/resnet18/spectral_safe/analysis/summary.csv`
- `experiments/resnet18/spectral_safe/analysis/confusion_matrix.csv`
- `experiments/resnet18/spectral_safe/analysis/class_metrics.csv`
- `experiments/resnet18/spectral_safe/analysis/misclassified_samples.csv`

### Per-class highlights

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `0` | 0.9832 | 1.0000 | 0.9915 |
| `1` | 0.9854 | 0.9802 | 0.9828 |
| `2` | 0.9765 | 0.9684 | 0.9724 |
| `3` | 0.9296 | 0.9487 | 0.9391 |

Observation:
- Class `3` is currently the weakest class.
- The largest confusion concentrations are between classes `1`, `2`, and `3`.

## Submission Artifact

The current best submission file has already been generated:

- `datasets/submissions/resnet18/spectral_safe/submission_resnet18.csv`

## Pipelines Implemented

The following pipelines are available through `run.py`:

- `prepare_data`
- `train_baseline`
- `compare_baselines`
- `analyze_validation`
- `predict_submission`
- `run_kfold_evaluation`

## Recommended Report Structure

1. Problem statement and dataset overview
2. Preprocessing decisions
3. Baseline comparison: `cnn` vs `resnet18`
4. Augmentation comparison for `resnet18`
5. Error analysis of the best holdout model
6. Final submission strategy and limitations

## Reproducible Commands

### Train the best holdout model

```bash
python run.py mode.pipeline_name=train_baseline model=resnet18 aug=spectral_safe runtime.device=cpu
```

### Analyze the best holdout model

```bash
python run.py mode.pipeline_name=analyze_validation model=resnet18 aug=spectral_safe
```

### Generate the final submission

```bash
python run.py mode.pipeline_name=predict_submission model=resnet18 aug=spectral_safe runtime.device=cpu
```

### Run stratified k-fold evaluation

```bash
python run.py mode.pipeline_name=run_kfold_evaluation model=resnet18 aug=spectral_safe train=kfold runtime.device=cpu
```

## Note on Cross-validation

The `stratified k-fold` evaluation pipeline is fully implemented, including:
- per-fold checkpoints
- per-fold histories
- per-fold validation predictions
- aggregate fold metrics
- out-of-fold predictions
- aggregate confusion matrix
- aggregate class metrics

Because full CPU execution is time-consuming, it should be launched as a dedicated long-running experiment when you are ready to extend the report beyond the holdout conclusion.
