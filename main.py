# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Verify that logging is working
logging.info("Logging is set up.")

# Create directories for saving outputs
os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)

# Import dataset
logging.info("Loading dataset")
heart_data = pd.read_csv('heart.csv')

# Cost matrix
tp = 0
fn = 10
tn = 0
fp = 1
cost_matrix = np.array([[tn, fp], [fn, tp]])

# Data Exploration
logging.info("Exploring dataset")
data = pd.read_csv('heart.csv')
logging.info(f"Dataset head:\n{data.head(5)}")

# Histograms for the target variable
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
labels = [f"{i}-{i+4}" for i in range(0, 100, 5)]
idk = heart_data.groupby([pd.cut(heart_data['Age'], bins=bins, labels=labels, right=False), "HeartDisease"], observed=True).size().unstack("HeartDisease")
idk.columns = idk.columns.map({0: "No", 1: "Yes"})
histogram = idk.plot.bar(stacked=True, title="Distribution of Age Groups by Heart Disease Presence")
plt.xlabel('Heart Disease')
plt.ylabel('Number of Patients')
plt.savefig("outputs/images/age_distribution_heart_disease.jpg")
plt.close()

# Select columns of interest
cols_of_interest = ['MaxHR', 'RestingBP', 'Cholesterol']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for col, ax in zip(cols_of_interest, axes.flatten()):
    heart_data[col].hist(ax=ax, bins=50)
    ax.set_title(f'Histogram of {col}')
    ax.set_xlabel(f'{col} (units)')
    ax.set_ylabel('Number of Patients')
plt.tight_layout()
plt.savefig("outputs/images/columns_histograms.jpg")
plt.close()

# Correlation matrix
heart_data_num = heart_data.select_dtypes(['number'])
f = plt.figure(figsize=(8, 8))
plt.matshow(heart_data_num.corr(), fignum=f.number)
plt.xticks(range(heart_data_num.shape[1]), heart_data_num.columns, fontsize=14, rotation=45)
plt.yticks(range(heart_data_num.shape[1]), heart_data_num.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig("outputs/images/correlation_matrix.jpg")
plt.close()

# Correlation with HeartDisease
correlation = pd.DataFrame(heart_data_num.corr()["HeartDisease"].abs().drop(["HeartDisease"]).sort_values(ascending=False))
logging.info(f"Correlation with HeartDisease:\n{correlation}")

# Train test split
logging.info("Splitting data into train and test sets")
X = heart_data.drop('HeartDisease', axis=1)
y = heart_data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
logging.info("Preprocessing data")
set_config(transform_output="pandas")
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], verbose_feature_names_out=False)
preprocessor.fit(X_train)
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Classifier 1: Decision Tree
logging.info("Training Decision Tree classifier")
param_grid_dt = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_dt.fit(X_train_preprocessed, y_train)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_preprocessed, y_train)
dt_predictions = dt_model.predict(X_test_preprocessed)

# Classifier 2: Random Forest
logging.info("Training Random Forest classifier")
param_grid_rf = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train_preprocessed, y_train)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_preprocessed, y_train)
rf_predictions = rf_model.predict(X_test_preprocessed)

# Classifier 3: Dummy Classifier
logging.info("Training Dummy classifier")
dummy_model = DummyClassifier(strategy="uniform")
dummy_model.fit(X_train_preprocessed, y_train)
dummy_predictions = dummy_model.predict(X_test_preprocessed)

# Evaluation
logging.info("Evaluating models")
logging.info("\nDecision Tree Classification Report:")
logging.info(classification_report(y_test, dt_predictions))
logging.info(f"Decision Tree Model Accuracy: {accuracy_score(y_test, dt_predictions)}")
dt_conf = confusion_matrix(y_test, dt_predictions)
ConfusionMatrixDisplay(dt_conf).plot()
plt.savefig("outputs/images/dt_confusion_matrix.jpg")
plt.close()
dt_cost = (dt_conf * cost_matrix).sum()
logging.info(f"Cost of Decision Tree: {dt_cost}")

logging.info("Random Forest Classification Report:")
logging.info(classification_report(y_test, rf_predictions))
logging.info(f"Random Forest Model Accuracy: {accuracy_score(y_test, rf_predictions)}")
rf_conf = confusion_matrix(y_test, rf_predictions)
ConfusionMatrixDisplay(rf_conf).plot()
plt.savefig("outputs/images/rf_confusion_matrix.jpg")
plt.close()
rf_cost = np.multiply(rf_conf, cost_matrix).sum()
logging.info(f"Cost of Random Forest: {rf_cost}")

logging.info("Dummy Classification Report:")
logging.info(classification_report(y_test, dummy_predictions))
logging.info(f"Dummy Model Accuracy: {accuracy_score(y_test, dummy_predictions)}")
dummy_conf = confusion_matrix(y_test, dummy_predictions)
ConfusionMatrixDisplay(dummy_conf).plot()
plt.savefig("outputs/images/dummy_confusion_matrix.jpg")
plt.close()
dummy_cost = np.multiply(dummy_conf, cost_matrix).sum()
logging.info(f"Cost of dummy classifier: {dummy_cost}")

# Explanation
plot_tree(dt_model, max_depth=3)
plt.savefig("outputs/images/decision_tree_plot.jpg")
plt.close()

importances = dt_model.feature_importances_
forest_importances = pd.Series(importances, index=dt_model.feature_names_in_)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
plt.savefig("outputs/images/decision_tree_importances.jpg")
plt.close()

importances = rf_model.feature_importances_
forest_importances = pd.Series(importances, index=rf_model.feature_names_in_)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
plt.savefig("outputs/images/random_forest_importances.jpg")
plt.close()

# Local explanation
logging.info("Local explanation for chosen instance")
chosen_instance = heart_data.iloc[68]
chosen_instance_preprocessed = preprocessor.transform(X).iloc[68:69]
dt_prediction = dt_model.predict(chosen_instance_preprocessed)
rf_prediction = rf_model.predict(chosen_instance_preprocessed)
logging.info(f"Decision Tree Prediction: {dt_prediction[0]}")
logging.info(f"Random Forest Prediction: {rf_prediction[0]}")
logging.info(f"Probability of Heart Failure (Decision Tree): {dt_model.predict_proba(chosen_instance_preprocessed)[0,1]*100}%")
logging.info(f"Probability of Heart Failure (Random Forest): {rf_model.predict_proba(chosen_instance_preprocessed)[0, 1]*100}%")
pd.DataFrame(chosen_instance_preprocessed).T

# Modify the preprocessed instance
logging.info("Modifying preprocessed instance")
instance_preprocessed = preprocessor.transform(X).iloc[68:69]
instance_preprocessed['Cholesterol'] = 5.0
dt_prediction = dt_model.predict(instance_preprocessed)
rf_prediction = rf_model.predict(instance_preprocessed)
logging.info(f"Modified Instance Prediction (Decision Tree): {dt_prediction[0]}")
logging.info(f"Modified Instance Prediction (Random Forest): {rf_prediction[0]}")
logging.info(f"Modified Instance Prediction (Decision Tree): {dt_model.predict_proba(instance_preprocessed)[0,1]*100}%")
logging.info(f"Modified Instance Prediction (Random Forest): {rf_model.predict_proba(instance_preprocessed)[0,1]*100}%")
pd.DataFrame(instance_preprocessed).T

# Export train and test data files as csv
logging.info("Exporting preprocessed train and test data to CSV")
X_train_preprocessed.to_csv("outputs/data/train.csv", index=False)
X_test_preprocessed.to_csv("outputs/data/test.csv", index=False)
