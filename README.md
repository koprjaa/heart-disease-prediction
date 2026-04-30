# 4IZ210 — Heart Disease Prediction

**Course project for 4IZ210 Machine Learning I, Prague University of Economics and Business: Jan Alexandr Kopřiva, David Hložek, Jakub Hermann, Ondřej Čech, Milan Tvrdík.**

![python](https://img.shields.io/badge/python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![license](https://img.shields.io/badge/license-MIT-A31F34?style=flat-square)
![status](https://img.shields.io/badge/status-complete-22863A?style=flat-square)
![jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?style=flat-square&logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.9-11557C?style=flat-square)

Two-part term project on 918 clinical records:

| Part | Approach | File |
|---|---|---|
| **1: Classification** | Decision Tree + Random Forest, asymmetric cost matrix | `heart_disease_prediction.ipynb` |
| **2: Clustering** | K-Means + Ward hierarchical, elbow + silhouette, PCA viz | `heart_disease_clustering.py` |

Both parts share the same dataset and asymmetric-cost philosophy `[[TN=0, FP=1], [FN=10, TP=0]]` — missing a diagnosis costs more than a false alarm.

## Part 1 — Classification

Decision Tree and Random Forest classifiers trained on 11 clinical features, evaluated against both plain accuracy and the cost-weighted matrix.

### Pipeline

```
heart.csv (918 rows × 12 cols)
  │
  ├── EDA — distributions, correlations, class balance
  │
  ├── Preprocessing (ColumnTransformer)
  │     • StandardScaler on 6 numeric
  │     • OneHotEncoder(handle_unknown='ignore') on 5 categorical
  │
  ├── Training
  │     • Train/test split 80/20, random_state=42
  │     • GridSearchCV hyperparameter exploration (3-fold CV)
  │     • Final models with default params (separates search from eval — avoids data leakage)
  │
  ├── Evaluation
  │     • Standard: accuracy, precision, recall, F1
  │     • Custom: total cost = sum(confusion_matrix * cost_matrix)
  │     • DummyClassifier as lower bound
  │
  └── Interpretation
        • Feature importance (Gini-based for trees)
        • Local predictions per patient
        • Counterfactual analysis ("what if cholesterol was X?")
```

### Run it

```bash
uv venv
uv pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

Execute cells top to bottom. Full pass takes ~2-3 minutes on a laptop (GridSearchCV is the bottleneck). 31 code cells, all self-contained.

## Part 2 — Clustering

Unsupervised analysis of the same 918 records — drop the `HeartDisease` label, ask whether natural clusters in feature space coincide with disease/no-disease split.

### Pipeline

```
heart.csv (918 rows × 12 cols)
  │
  ├── Preprocessing (ColumnTransformer)
  │     • MinMaxScaler on 6 numeric features
  │     • OneHotEncoder on 5 categorical features
  │     • HeartDisease removed (kept only for Rand-score evaluation)
  │     • No train/test split (unsupervised)
  │
  ├── K-Means
  │     • k=2 baseline (matches binary target)
  │     • KElbowVisualizer over k=2..15
  │       — best by inertia: k=6
  │       — best by silhouette: k=2
  │     • PCA-reduced 2D scatter visualisation
  │
  ├── Hierarchical (agglomerative)
  │     • Ward linkage
  │     • Dendrogram
  │     • fcluster cuts at multiple thresholds
  │
  └── Evaluation
        • Silhouette score (intrinsic)
        • Rand score vs. ground-truth HeartDisease (extrinsic)
        • Per-cluster aggregations (mean numeric, mode categorical) for interpretation
```

### Customization (assignment-specific)

| Parameter | Value |
|---|---|
| Target attribute (held out for eval) | `HeartDisease` |
| Instance of interest | row 69 (`heart_data.iloc[68]`) |
| Attribute of interest | `Cholesterol` |

### Run it

```bash
uv pip install pandas numpy scikit-learn scipy matplotlib yellowbrick
python heart_disease_clustering.py
```

Split into `# %%` cells — best run interactively (Jupyter, VS Code Python, Spyder).

## Dataset

`heart.csv` — 918 samples, 12 columns. Likely derived from the UCI Heart Disease dataset (Cleveland / Hungary / Switzerland / Long Beach combined), but provenance isn't documented.

| Column | Type | Meaning |
|---|---|---|
| Age | numeric | years |
| Sex | cat | M / F |
| ChestPainType | cat | TA / ATA / NAP / ASY |
| RestingBP | numeric | mm Hg |
| Cholesterol | numeric | mg/dl (some zeros — likely missing) |
| FastingBS | binary | > 120 mg/dl |
| RestingECG | cat | Normal / ST / LVH |
| MaxHR | numeric | bpm |
| ExerciseAngina | binary | Y / N |
| Oldpeak | numeric | ST depression |
| ST_Slope | cat | Up / Flat / Down |
| **HeartDisease** | binary | **target** (0/1) |

## Counterfactual example

```python
instance = preprocessor.transform(X).iloc[68:69]
print(f"DT: P(disease) = {dt_model.predict_proba(instance)[0,1]:.2%}")
print(f"RF: P(disease) = {rf_model.predict_proba(instance)[0,1]:.2%}")

# what if cholesterol dropped?
instance_mod = instance.copy()
instance_mod['Cholesterol'] = 5.0  # scaled
print(f"RF with low chol: {rf_model.predict_proba(instance_mod)[0,1]:.2%}")
```

## Honest limitations

- **Grid search results aren't used** — best params are printed but final models use defaults (intentional, to avoid nested-CV leakage; documented in the notebook).
- **Single train/test split** — no repeated runs, no k-fold on the final eval. Results may not generalise.
- **Cost ratio is arbitrary** — 10:1 is illustrative, not clinically-calibrated.
- **Cholesterol = 0 rows** aren't handled — they're biologically impossible and probably missing values.
- **No model persistence** — re-run the notebook to re-train.
- **Default 0.5 probability threshold** — cost-sensitive threshold tuning would improve the cost score measurably.

## License

[MIT](LICENSE)
