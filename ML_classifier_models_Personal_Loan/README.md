# Personal Loan Acceptance Prediction

An exploratory machine learning project that predicts whether a bank customer will accept a personal loan offer. The notebook walks through data exploration, feature screening, preprocessing, model fitting, hyperparameter tuning and comparison of three classifiers.

**The target is loan offer acceptance—not bank approval, creditworthiness or repayment default.**

## Notebook and dataset

The analysis is implemented in [Personal_loan_data_analysis.ipynb](Personal_loan_data_analysis.ipynb). Code is organised into sequential notebook cells rather than separate Python scripts.

Download the dataset from [Personal Loan Modeling on Kaggle](https://www.kaggle.com/datasets/teertha/personal-loan-modeling). Place `Bank_Personal_Loan_Modelling.csv` in the notebook's working directory, or update the path in the first cell.

The analysed dataset contains **5,000 customers and 14 columns**, with no missing values reported in the saved initial inspection.

| Target: `Personal Loan` | Customers | Proportion |
|---|---:|---:|
| 0: Did not accept | 4,520 | 90.4% |
| 1: Accepted | 480 | 9.6% |

Always predicting non-acceptance achieves 90.4% accuracy, so accuracy alone is inadequate for this task.

### Data dictionary

| Column | Description |
|---|---|
| `ID` | Customer identifier; excluded from modelling |
| `Age` | Age in years |
| `Experience` | Professional experience in years; negative values require investigation |
| `Income` | Annual income in thousands of US dollars |
| `ZIP Code` | Geographic category; excluded from the modelling baseline |
| `Family` | Family size |
| `CCAvg` | Average monthly credit card spending in thousands of US dollars |
| `Education` | Education category: 1 = undergraduate, 2 = graduate, 3 = advanced/professional |
| `Mortgage` | Mortgage value in thousands of US dollars |
| `Personal Loan` | Target: 1 = accepted the offer, 0 = did not accept |
| `Securities Account` | Securities account ownership indicator |
| `CD Account` | Certificate of deposit account ownership indicator |
| `Online` | Online banking usage indicator |
| `CreditCard` | Ownership of a credit card issued by the bank |

## How to run

1. Download the notebook and dataset into the same working directory.
2. Create and activate a Python environment if desired.
3. Install dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

4. Start JupyterLab:

```bash
jupyter lab
```

5. Open `Personal_loan_data_analysis.ipynb`, select the appropriate Python kernel, and run the cells from top to bottom. Use **Restart Kernel and Run All Cells** for a clean run.

Cells share variables, so running later cells before earlier ones can cause `NameError` exceptions. Hyperparameter searches use multiple CPU cores and may take several minutes. Set `n_jobs=1` in the relevant calls if resource usage is too high.

The notebook does not pin dependency versions. Saved numerical results may differ slightly across package versions; record your environment versions when reproducing the analysis.

## Analysis workflow

| Stage | What the code does | Main outputs |
|---|---|---|
| Data loading | Reads the CSV and inspects shape, types and target counts | Data preview and class distribution |
| Income exploration | Compares income distributions and acceptance rates by income band | Summary tables, boxplot and bar chart |
| All-feature EDA | Loops through predictors and compares them with acceptance | Distribution plots and acceptance rates with group sizes |
| Pearson screening | Creates the train/test split and examines linear associations on training data | Feature ranking and correlation heatmap |
| Mutual information | Estimates individual dependence, including nonlinear relationships | MI ranking and bar chart |
| Logistic regression | Fits a preprocessed linear baseline and performs 10-fold CV | Fold scores, confusion matrices and probability plots |
| Decision tree | Tunes tree complexity with grid search | Best settings, metrics, confusion matrices and tree diagram |
| Random forest | Tunes an ensemble of trees | Best settings, metrics, confusion matrices and permutation importance |
| Model comparison | Evaluates the three fixed configurations on identical CV folds | Metric tables and comparison charts |

In the EDA loop, continuous features are generally divided into quantile groups. Mortgage values of zero are separated from positive mortgage values. Only the 15 most frequent ZIP codes are plotted. These groups are for visual analysis and are not the inputs used by the final models.

## Feature screening and selection

Pearson correlation measures linear association; with a binary target and continuous predictor, it equals point-biserial correlation. Mutual information can also detect nonlinear dependence, but has no direction and is not bounded by one.

The modelling shortlist is:

```python
selected_features = [
    "Income",
    "CCAvg",
    "CD Account",
    "Mortgage",
    "Education"
]
```

| Selected feature | Pearson correlation with target | Mutual information |
|---|---:|---:|
| Income | 0.497 | 0.1305 |
| CCAvg | 0.366 | 0.0917 |
| CD Account | 0.337 | 0.0341 |
| Mortgage | 0.140 | 0.0223 |
| Education | Not assessed using numeric category codes | 0.0121 |

`ID` is removed as an identifier and `ZIP Code` is excluded from the baseline. Age, Experience, Family, Securities Account, Online and CreditCard are omitted from the final shortlist because of their weaker individual screening results. This is a provisional selection: low individual association does not establish that a feature is useless in combination with others.

Negative Experience values are flagged during EDA, treated as missing for correlation/MI analysis, and imputed for MI. Experience is not included in the final model inputs.

## Data split and preprocessing

- **80% training / 20% test:** 4,000 and 1,000 customers, respectively.
- The split uses `stratify=y` and `random_state=42`.
- Cross-validation uses `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` on the training set.
- Each CV fit trains on approximately 3,600 customers and validates on 400.
- `Pipeline` and `ColumnTransformer` fit model preprocessing within each training fold.

| Feature type | Logistic regression | Decision tree / random forest |
|---|---|---|
| Income, CCAvg, Mortgage | Median imputation and standardisation | Median imputation; no scaling |
| Education | Most-frequent imputation; one-hot encoding with first category dropped | Most-frequent imputation; full one-hot encoding |
| CD Account | Most-frequent imputation; retained as binary | Same |

The classifiers use their default class weighting; the notebook does not apply oversampling or undersampling. Class-based metrics use the default decision rule, corresponding to a probability boundary of 0.5. ROC-AUC and average precision use probability scores.

## Models and tuning

### Logistic regression

A regularised baseline that models acceptance probability through a linear combination of inputs in log-odds space. The notebook uses `solver="lbfgs"`, `C=1.0` and `max_iter=2000`. Its hyperparameters are not tuned.

### Decision tree

Learns nonlinear if–then rules and feature interactions. Grid search evaluates 18 combinations:

| Hyperparameter | Values searched |
|---|---|
| `max_depth` | 3, 5, 8 |
| `min_samples_leaf` | 5, 10, 20 |
| `min_samples_split` | 2, 20 |

Saved best settings: **depth 5, minimum leaf size 10, minimum split size 2**. The fitted tree has 18 leaves and 35 nodes. Some settings can produce identical trees; the saved output includes tied combinations.

### Random forest

Averages predictions from multiple trees trained with bootstrap sampling. Grid search evaluates 16 combinations:

| Hyperparameter | Values searched |
|---|---|
| `n_estimators` | 100, 300 |
| `max_depth` | 5, None (unlimited) |
| `min_samples_leaf` | 1, 5 |
| `max_features` | `"sqrt"`, 1.0 (all transformed features) |

Saved best settings: **300 trees, depth 5, minimum leaf size 5 and `max_features=1.0`**. Grid search parallelises fits while each forest uses `n_jobs=1` to avoid nested parallelism.

Both searches select the highest mean validation **average precision** and automatically refit the selected pipeline on all training observations. In scikit-learn CV output, `test_` denotes validation folds, not the separate 1,000-customer test set.

## Evaluation metrics

The positive class is acceptance (`Personal Loan = 1`).

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Precision | Fraction of predicted acceptors who actually accepted |
| Recall | Fraction of actual acceptors identified |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Ranking discrimination across thresholds |
| Average precision (AP) | Precision averaged over increases in recall; useful for the minority class |

AP is the primary tuning metric. A random ranking has expected AP approximately equal to the acceptance prevalence (0.096), whereas its ROC-AUC is approximately 0.5. AP is not identical to trapezoidal area under the precision–recall curve.

## Results recorded in the notebook

These tables reproduce the uploaded notebook's saved outputs; they are not a new execution or an independent validation.

### Mean 10-fold validation scores

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.954 | 0.863 | 0.626 | 0.723 | 0.952 | 0.821 |
| Decision tree | 0.968 | 0.896 | 0.761 | 0.820 | 0.980 | 0.899 |
| Random forest | 0.972 | 0.923 | 0.779 | 0.841 | 0.985 | 0.918 |

### Held-out test scores

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.957 | 0.863 | 0.656 | 0.746 | 0.946 | 0.836 |
| Decision tree | 0.973 | 0.888 | 0.823 | 0.854 | 0.982 | 0.903 |
| Random forest | 0.973 | 0.888 | 0.823 | 0.854 | 0.986 | 0.925 |

Random forest achieved the highest recorded validation AP. The tree and forest have identical rounded test accuracy, precision, recall and F1, but different ranking scores because their predicted probabilities differ.

Test-set permutation importance for the forest ranks Income and Education highest. This illustrates why weak individual screening scores do not necessarily imply low importance within a fitted model. Importance is model-dependent and does not establish causation.

## Important limitations

- Initial EDA uses the full dataset before splitting. The eventual test set is therefore not completely unseen during exploration. A stricter workflow would split before target-aware EDA.
- Feature screening uses all training data before CV. Repeating selection within CV would provide a more rigorous assessment of the complete selection process.
- Tree and forest hyperparameters are tuned and compared using the same folds. Their winning CV scores can be optimistic; nested CV is appropriate for evaluating the complete tuning procedure.
- Logistic regression is untuned while the tree models are tuned, so the comparison reflects these specific configurations and search budgets.
- All three models are evaluated on the same test set. Avoid further model, feature or threshold decisions based on these test results; use new independent data for a fresh final assessment if needed.
- Low univariate correlation or MI can overlook interactions. Comparing the shortlist with a broader feature set is a useful future experiment.
- Observed associations do not demonstrate causation, probability calibration or performance on future customers. Confirm that input attributes would be available before the campaign response when applying this approach elsewhere.

## Possible extensions

- Add a majority-class dummy baseline and compare additional feature subsets.
- Tune logistic regression regularisation within CV.
- Use nested CV and fold-specific feature selection.
- Inspect precision–recall curves and choose a contact threshold using validation data and campaign costs.
- Assess calibration and evaluate on an independent campaign dataset.
- Pin dependencies and export the final fitted pipeline for reproducible inference.

## Dataset acknowledgement

Dataset: [Personal Loan Modeling, Kaggle](https://www.kaggle.com/datasets/teertha/personal-loan-modeling). Consult the dataset page for its terms before redistributing data. This repository documents an educational classification analysis.
