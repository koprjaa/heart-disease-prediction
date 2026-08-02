# 4IZ210

Heart disease prediction on 918 clinical records. Course project for 4IZ210 Machine Learning I at Prague University of Economics and Business.

![python](https://img.shields.io/badge/python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![license](https://img.shields.io/badge/license-MIT-A31F34?style=flat-square)
![status](https://img.shields.io/badge/status-complete-22863A?style=flat-square)
![jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

Authors: Jan Alexandr Kopřiva, David Hložek, Jakub Hermann, Ondřej Čech, Milan Tvrdík.

## What it does

The project has two parts on the same dataset.

| Part | Method | File |
|---|---|---|
| Classification | Decision tree and random forest, with an asymmetric cost matrix | `heart_disease_prediction.ipynb` |
| Clustering | K-means and Ward hierarchical, elbow and silhouette, PCA plot | `heart_disease_clustering.py` |

Both parts use the cost matrix `[[TN=0, FP=1], [FN=10, TP=0]]`. A missed diagnosis costs ten times more than a false alarm.

## Install

```bash
uv venv
uv pip install -r requirements.txt
```

## Use

Classification:

```bash
jupyter notebook heart_disease_prediction.ipynb
```

Run the cells from top to bottom. The notebook holds 31 self contained code cells. A full pass takes two to three minutes on a laptop. `GridSearchCV` takes most of that time.

Clustering:

```bash
python heart_disease_clustering.py
```

The file is split into `# %%` cells. Run it in Jupyter, VS Code, or Spyder.

## Classification pipeline

```
heart.csv (918 rows, 12 columns)
  |
  +-- Exploration: distributions, correlations, class balance
  |
  +-- Preprocessing with ColumnTransformer
  |     StandardScaler on 6 numeric columns
  |     OneHotEncoder(handle_unknown='ignore') on 5 categorical columns
  |
  +-- Training
  |     Train and test split 80/20, random_state=42
  |     GridSearchCV with 3-fold cross validation
  |     Final models use default parameters
  |
  +-- Evaluation
  |     Accuracy, precision, recall, F1
  |     Total cost as the sum of confusion_matrix * cost_matrix
  |     DummyClassifier as the lower bound
  |
  +-- Interpretation
        Gini feature importance
        Per patient predictions
        Counterfactual analysis
```

## Clustering pipeline

```
heart.csv (918 rows, 12 columns)
  |
  +-- Preprocessing with ColumnTransformer
  |     MinMaxScaler on 6 numeric columns
  |     OneHotEncoder on 5 categorical columns
  |     HeartDisease removed, kept only to compute the Rand score
  |     No train and test split
  |
  +-- K-means
  |     k=2 baseline, which matches the binary target
  |     KElbowVisualizer over k=2 to 15
  |       best by inertia: k=6
  |       best by silhouette: k=2
  |     PCA reduced 2D scatter plot
  |
  +-- Hierarchical, agglomerative
  |     Ward linkage, dendrogram, fcluster cuts at several thresholds
  |
  +-- Evaluation
        Silhouette score
        Rand score against the HeartDisease label
        Per cluster aggregates for interpretation
```

The assignment fixed three parameters: `HeartDisease` as the held out target, row 69 as the instance of interest, and `Cholesterol` as the attribute of interest.

## Dataset

`heart.csv` holds 918 samples and 12 columns. The data appears to come from the UCI Heart Disease dataset, which combines the Cleveland, Hungary, Switzerland, and Long Beach sets. The provenance is not documented.

| Column | Type | Meaning |
|---|---|---|
| Age | numeric | Years |
| Sex | categorical | M or F |
| ChestPainType | categorical | TA, ATA, NAP, ASY |
| RestingBP | numeric | mm Hg |
| Cholesterol | numeric | mg/dl. Some zero values, probably missing. |
| FastingBS | binary | Above 120 mg/dl |
| RestingECG | categorical | Normal, ST, LVH |
| MaxHR | numeric | Beats per minute |
| ExerciseAngina | binary | Y or N |
| Oldpeak | numeric | ST depression |
| ST_Slope | categorical | Up, Flat, Down |
| HeartDisease | binary | Target, 0 or 1 |

## Limits

- The grid search results are printed but not used. The final models keep the default parameters. This avoids leakage from nested cross validation and the notebook says so.
- One train and test split. No repeated runs and no k-fold on the final evaluation. The numbers may not generalize.
- The 10 to 1 cost ratio illustrates the method. It is not calibrated against clinical data.
- Rows with `Cholesterol = 0` stay in the data. That value is not biologically possible and probably means a missing value.
- The notebook saves no model. Every run trains again.
- The probability threshold stays at the default 0.5. Tuning it against the cost matrix would improve the cost score.

## License

[MIT](LICENSE)
