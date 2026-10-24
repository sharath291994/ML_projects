# Methane Steam Reforming: Regression Model Comparison

A Python machine learning project that explores relationships between operating conditions, surface coverages, and turnover frequency (TOF), and compares four regression models.

**Data provenance:** The dataset contains 10,000 datapoints on MSR TOFs (s⁻¹) at different operating conditions obtained from a mean-field microkinetic model. 

## Dataset

Input file: `methane_steam_reforming_data.csv`

| Column | Meaning | Units |
|---|---|---|
| `temperature_K` | Temperature | K |
| `pressure_bar` | Total pressure | bar |
| `coverage_CH` | Fractional CH coverage | Dimensionless |
| `coverage_C` | Fractional C coverage | Dimensionless |
| `coverage_OH` | Fractional OH coverage | Dimensionless |
| `coverage_O` | Fractional O coverage | Dimensionless |
| `coverage_CO` | Fractional CO coverage | Dimensionless |
| `TOF_s-1` | Turnover frequency; regression target | s⁻¹ |

- 10,000 rows and eight numeric columns: seven features and one target.
- Temperature sampled over 500–1,500 K.
- Pressure sampled logarithmically over 0.1–100 bar.
- Assumed steam-to-methane ratio: 3; it is fixed and is not a feature.
- The five coverages sum to less than one; the remainder represents vacant sites in the illustrative generation scheme.
- The supplied dataset contains no missing values.

The generation scheme uses an activated temperature dependence, a saturating pressure contribution, coverage interactions, and approximately 8% multiplicative noise. Its parameters are illustrative rather than calibrated physical constants.

## Analysis workflow

### 1. Feature association analysis

Calculate Pearson correlations and mutual information (MI) for both TOF and `log10(TOF)`.

- **Pearson correlation:** signed linear association, ranging from −1 to +1.
- **Mutual information:** estimates dependence, including nonlinear relationships; reported in nats, with no direction or fixed upper bound.

This stage uses an 80/20 split, with association analysis restricted to the 8,000-row training subset. The later regression cells use a separate 10-fold procedure on the full dataset; the earlier 2,000-row subset is therefore **not an independent final holdout** for those cells.

Association scores do not establish causation or independent feature effects. Temperature and coverages are correlated, and coverages are constrained by the available sites. A negative marginal correlation does not by itself establish inhibition.

### 2. Regression models

| Model | Preprocessing | Purpose |
|---|---|---|
| Linear regression | Standardise features | Simple linear baseline |
| Decision tree | No scaling | Capture nonlinear relationships using individual tree splits |
| Random forest | No scaling | Combine predictions from multiple bootstrap-trained trees |
| ANN (`MLPRegressor`) | Standardise features and target | Learn nonlinear relationships with a feedforward neural network |

The default setting is `USE_LOG_TARGET = True`. Models therefore predict `log10(TOF)` and their reported RMSE values are in log10 units. All models must use the same target setting for a valid comparison.

### 3. Ten-fold evaluation and tuning

Use shuffled `KFold(n_splits=10, random_state=42)`:

1. Train on 9,000 rows and test on 1,000 rows in each fold.
2. Repeat until every row has appeared in a test fold once.
3. For each hyperparameter combination, average train and test RMSE across the ten folds.
4. Select the configuration with the lowest average test RMSE.

Identical folds are reused across parameter combinations and models. Scaling is fitted within each outer training fold, without access to the outer test rows.

The simplified tuning workflow uses the same folds for parameter selection and reporting. Its winning scores are tuning estimates and can be optimistic. It does not use nested cross-validation or a separate final test set.

### 4. Hyperparameter grids

| Model | Hyperparameters tested | Combinations |
|---|---|---:|
| Decision tree | `max_depth=[5, 10, None]`; `min_samples_leaf=[1, 5, 20]` | 9 |
| Random forest | `n_estimators=[100, 200]`; `max_depth=[10, None]`; `min_samples_leaf=[1, 5]` | 8 |
| ANN | `hidden_layer_sizes=[(32,), (64, 32)]`; `alpha=[0.0001, 0.01]`; `learning_rate_init=[0.001, 0.003]` | 8 |

`None` allows unrestricted maximum depth, subject to other tree stopping conditions. Random forest considers all seven features at each split (`max_features=1.0`).

The ANN uses ReLU activation, Adam optimisation, a maximum of 1,000 epochs, and early stopping. It reserves 10% of each outer training subset for internal monitoring: approximately 8,100 rows update weights, 900 monitor stopping, and the outer 1,000 test rows remain excluded. Reported ANN training RMSE includes the internal monitoring rows. Feature and target scalers use the outer training subset, including those monitoring rows.

### 5. Final model comparison

The comparison cell loads the selected settings from each model's saved `hyperparameter_results.csv`, evaluates those configurations on identical ten-fold splits, and ranks models by average test RMSE. It does not repeat the parameter searches.

## Metrics and interpretation

| Metric | Meaning | Preferred value |
|---|---|---|
| RMSE | Square root of mean squared prediction error | Lower |
| R² | Variance explained relative to a constant-mean baseline; can be negative | Higher |
| Test minus train RMSE | Difference between average test and training error | Interpret alongside absolute errors |
| Fit time | Approximate training time per fold | Lower when accuracy is comparable |

With the log target, an individual absolute error of 1 log10 unit corresponds to a factor of 10 in TOF. Log-scale RMSE is not an error in s⁻¹. Back-transform predictions with `10 ** predicted_log_tof`; this does not automatically give bias-corrected conditional mean TOF predictions.

A large train/test error gap can indicate overfitting. A small gap alone does not establish accuracy. Fold standard deviations describe variability; they are not confidence intervals or significance tests.

The preferred accuracy model is the one with the lowest average test RMSE among the tested configurations. When errors are similar, consider simplicity, fit time, and deployment needs. The analysis cell generates its recommendation from the computed results rather than assuming an ensemble or ANN will win.

## Parity plots

The diagonal line represents perfect agreement between actual and predicted targets.

- Individual model plots show training and test predictions for **fold 1 only**, with that fold's metrics.
- The comparison figure shows **fold 1 test predictions** for all four models, using the same 1,000 held-out rows and common axis limits.
- These plot metrics are not the averages across all ten folds.

### Linear regression — fold 1

![Linear regression training and test parity for fold 1](linear_regression_outputs/linear_regression_fold1_parity.png)

### Decision tree — fold 1

![Decision tree training and test parity for fold 1](decision_tree_outputs/selected_parameters_fold1_parity.png)

### Random forest — fold 1

![Random forest training and test parity for fold 1](random_forest_outputs/selected_parameters_fold1_parity.png)

### Artificial neural network — fold 1

![ANN training and test parity for fold 1](ann_outputs/selected_parameters_fold1_parity.png)

### Comparison of all models — fold 1 test data

![Test parity comparison of all four models for fold 1](model_comparison_outputs/fold1_test_parity_comparison.png)

### Average training and test RMSE

![Average training and test RMSE across ten folds](model_comparison_outputs/model_rmse_comparison.png)

These images use repository-relative links. Upload the PNG files in the output folders shown above, with this README in the repository root. Filenames and capitalisation must match exactly. Run the notebook cells first if the images have not yet been generated.

## Setup and execution

Requires Python 3 and the following packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

1. Place the CSV and project notebook in the same working directory.
2. Open the notebook with Jupyter:

   ```bash
   jupyter notebook
   ```

3. Run cells in this order:
   - Imports and feature association analysis.
   - Linear regression.
   - Decision tree tuning and parity plots.
   - Random forest tuning and parity plots.
   - ANN tuning and parity plots.
   - Final model comparison and analysis.

4. Keep `USE_LOG_TARGET` consistent across the model cells.
5. Run tuning cells before the final comparison: the comparison reads their saved CSV files.

Training runs on a CPU; a GPU is not required. Random forest uses available CPU cores via `n_jobs=-1`. ANN and random forest tuning can take several minutes depending on hardware.

## Generated outputs

Paths are relative to the notebook's working directory.

| Directory | Important files | Contents |
|---|---|---|
| `feature_analysis_outputs/` | `pearson_correlation_heatmap.png` | Feature/target correlation matrix |
| `feature_analysis_outputs/` | `pearson_feature_correlations.png` | Feature–target Pearson correlations |
| `feature_analysis_outputs/` | `mutual_information_scores.png` | Feature–target MI estimates |
| `feature_analysis_outputs/` | `feature_association_results.csv` | Numerical association results |
| `linear_regression_outputs/` | `cv_fold_metrics.csv`, `cv_metrics_summary.csv` | Per-fold and summary regression metrics |
| `linear_regression_outputs/` | `linear_regression_fold1_parity.png` | Fold 1 training/test parity |
| `linear_regression_outputs/` | `out_of_fold_test_predictions.csv` | Held-out predictions for all rows |
| `decision_tree_outputs/` | `hyperparameter_results.csv` | Ranked tuning configurations and average RMSE |
| `decision_tree_outputs/` | `selected_parameters_fold1_parity.png` | Fold 1 training/test parity |
| `random_forest_outputs/` | `hyperparameter_results.csv` | Ranked tuning configurations and average RMSE |
| `random_forest_outputs/` | `selected_parameters_fold1_parity.png` | Fold 1 training/test parity |
| `ann_outputs/` | `hyperparameter_results.csv` | Ranked tuning configurations and average RMSE |
| `ann_outputs/` | `selected_parameters_fold1_parity.png` | Fold 1 training/test parity |
| `model_comparison_outputs/` | `all_model_fold_metrics.csv` | Per-fold metrics for all models |
| `model_comparison_outputs/` | `model_comparison_summary.csv` | Ranked average metrics and fit times |
| `model_comparison_outputs/` | `model_rmse_comparison.png` | Average train/test RMSE with test fold SD |
| `model_comparison_outputs/` | `fold1_test_parity_comparison.png` | Four-model test parity comparison |
| `model_comparison_outputs/` | `model_comparison_analysis.txt` | Automatically generated interpretation and recommendation |

The supplied model cells fit final estimators in memory; they do not save trained model binaries. Rerunning a cell can overwrite its output files.

## Reproducibility and limitations

- Random seeds are fixed at 42 where applicable. Keep row ordering unchanged to reproduce the same folds.
- Results can vary with package versions, numerical libraries, and hardware. Record environment versions when reporting results, for example with `pip freeze > requirements.txt`.
- Random-fold results primarily assess interpolation within the sampled distribution. They do not demonstrate extrapolation to new operating ranges.
- Surface coverages are supplied as inputs. Predicting TOF from operating conditions alone would require another model or a method to obtain these coverages.
- Selected hyperparameters are optimal only among the combinations tested.
- For an independent final assessment, reserve a holdout before feature selection/tuning or use nested cross-validation.
- No permutation importance is performed in this workflow.
