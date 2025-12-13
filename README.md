![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-blue?style=flat-square)

# Heart Disease Prediction

Binary classification of heart disease using Decision Tree and Random Forest classifiers with asymmetric cost evaluation.

## Motivation

This project was developed for the course 4IZ210 (Machine Learning I) at Prague University of Economics and Business. It demonstrates a complete ML pipeline from exploratory analysis through model interpretation. The asymmetric cost matrix addresses a real clinical concern: missing a disease diagnosis has greater consequences than a false alarm.

## What This Project Does

Predicts presence of heart disease (binary classification) from 11 clinical features. Compares two tree-based models against a random baseline using both accuracy and cost-weighted evaluation.

Key outputs:
- Model comparison (Decision Tree vs Random Forest vs Dummy baseline)
- Feature importance rankings
- Confusion matrices with cost analysis
- Local prediction explanations with counterfactual analysis

## Architecture

Single Jupyter notebook pipeline with seven stages:

```
Data Loading -> EDA -> Preprocessing -> Training -> Evaluation -> Interpretation -> Export
```

**Preprocessing Pipeline:**
- `ColumnTransformer` splits numeric and categorical features
- Numeric: `StandardScaler` (6 features)
- Categorical: `OneHotEncoder` (5 features -> expanded)
- Output: pandas DataFrames with preserved feature names

**Training:**
- `GridSearchCV` explores hyperparameter space (3-fold CV)
- Final models trained with default parameters
- Train/test split: 80/20, fixed seed

**Evaluation:**
- Standard metrics: accuracy, precision, recall, F1
- Custom cost matrix: `[[TN=0, FP=1], [FN=10, TP=0]]`
- Total cost = sum of (confusion matrix * cost matrix)

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Data manipulation | pandas | 2.2.2 |
| Numerical operations | numpy | >=1.24.0 |
| ML models and preprocessing | scikit-learn | 1.5.1 |
| Visualization | matplotlib | 3.9.1 |
| Environment | Jupyter Notebook | >=7.0.0 |

## Data Sources

**Dataset:** `heart.csv` (918 samples, 12 columns)

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age in years |
| Sex | Categorical | M/F |
| ChestPainType | Categorical | TA, ATA, NAP, ASY |
| RestingBP | Numeric | Resting blood pressure (mm Hg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dl) |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dl |
| RestingECG | Categorical | Normal, ST, LVH |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Binary | Exercise-induced angina (Y/N) |
| Oldpeak | Numeric | ST depression |
| ST_Slope | Categorical | Up, Flat, Down |
| HeartDisease | Binary | Target variable (0/1) |

**Note:** The original data source is not documented in the repository.

## Key Design Decisions

**1. Asymmetric cost matrix (FN=10, FP=1)**
In clinical screening, a missed diagnosis (false negative) leads to untreated disease. A false positive leads to additional testing. The 10:1 ratio reflects this asymmetry, though the exact value is domain-dependent.

**2. DummyClassifier as baseline**
Uses uniform random predictions to establish a lower bound. Any model that does not significantly outperform random guessing provides no value.

**3. Default hyperparameters for final models**
GridSearchCV is run but final models use defaults. This separates hyperparameter exploration from the main evaluation, avoiding data leakage from improper CV nesting.

**4. StandardScaler for numeric features**
Tree-based models are scale-invariant. Scaling is applied for consistency with other potential classifiers and does not harm tree performance.

**5. OneHotEncoder with handle_unknown='ignore'**
Prevents errors if test data contains unseen categories. Returns zero vector for unknown values.

**6. Fixed random state (42)**
Ensures reproducibility of train/test split and model training.

## Limitations

1. **Grid search results unused:** Best parameters from GridSearchCV are printed but not applied to final models. This is intentional (see design decisions) but may cause confusion.

2. **No nested cross-validation:** Hyperparameter search and final evaluation use overlapping data. Results may be optimistically biased.

3. **Single train/test split:** No repeated random splits or k-fold CV on final evaluation. Results may not generalize.

4. **Cost matrix values are arbitrary:** The 10:1 ratio is illustrative. Real clinical applications require domain expert input.

5. **No model persistence:** Trained models are not saved. Re-running the notebook is required for predictions.

6. **Dataset provenance unclear:** The source of `heart.csv` is not documented. Likely derived from UCI Heart Disease dataset but not verified.

7. **No threshold optimization:** Default 0.5 probability threshold is used. Cost-sensitive classification could benefit from threshold tuning.

8. **Cholesterol zeros not addressed:** Some records have Cholesterol=0, which is biologically impossible. These are likely missing values.

## How to Run

**Requirements:** Python 3.8+

```bash
# Clone repository
git clone https://github.com/your-username/4IZ210-heart-disease-prediction.git
cd 4IZ210-heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook heart_disease_prediction.ipynb
```

Execute cells sequentially. Full run takes approximately 2-3 minutes (GridSearchCV is the bottleneck).

## Example Usage

The notebook includes local prediction explanations. For a single patient:

```python
# After running preprocessing
instance = preprocessor.transform(X).iloc[68:69]

# Get probability of disease
dt_proba = dt_model.predict_proba(instance)[0, 1]
rf_proba = rf_model.predict_proba(instance)[0, 1]

print(f"Decision Tree: P(disease) = {dt_proba:.2%}")
print(f"Random Forest: P(disease) = {rf_proba:.2%}")
```

Counterfactual analysis (what-if):

```python
# Modify a feature and observe prediction change
instance_mod = instance.copy()
instance_mod['Cholesterol'] = 5.0  # Normalized value

rf_proba_mod = rf_model.predict_proba(instance_mod)[0, 1]
```

## Future Improvements

- Apply best GridSearchCV parameters to final models
- Implement nested cross-validation for unbiased evaluation
- Add threshold optimization based on cost matrix
- Handle Cholesterol=0 as missing data
- Persist models with joblib for deployment
- Add SHAP values for global interpretability
- Document dataset source and preprocessing steps

## Author

Jan Alexandr Kopriva  
jan.alexandr.kopriva@gmail.com

Co-authors (course project): David Hlozek, Jakub Hermann, Ondrej Cech, Milan Tvrdik

## License

MIT License. See [LICENSE](LICENSE) for details.
