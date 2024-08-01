
# Heart Disease Prediction Project

## Introduction
This project aims to predict the presence of heart disease in patients using various machine learning classifiers. The dataset used for this project is the Heart Disease dataset, which includes several features related to patient health metrics.

This project was developed as part of the 4IZ210 course.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributors](#contributors)
- [License](#license)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/koprjaa/heart-disease-prediction.git
   ```
2. Navigate to the project directory:
   ```sh
   cd heart-disease-prediction
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
The provided script `main.py` performs the following tasks:

1. **Import Libraries:** Imports necessary libraries and modules such as pandas, matplotlib, numpy, and various scikit-learn components.
2. **Load Dataset:** Loads the heart disease dataset from a CSV file.
3. **Data Exploration:** Performs initial data exploration including displaying the first few rows and visualizing distributions of the target variable.
4. **Histograms:** Generates histograms for selected columns to understand their distributions.
5. **Correlation Matrix:** Computes and displays a correlation matrix for the numerical features in the dataset.
6. **Train-Test Split:** Splits the dataset into training and testing sets.
7. **Preprocessing:** Preprocesses the data using scaling for numerical features and one-hot encoding for categorical features.
8. **Model Training and Evaluation:**
   - **Decision Tree Classifier:** Trains a Decision Tree model, performs hyperparameter tuning, evaluates the model, and calculates a cost matrix.
   - **Random Forest Classifier:** Trains a Random Forest model, performs hyperparameter tuning, evaluates the model, and calculates a cost matrix.
   - **Dummy Classifier:** Trains a Dummy model for baseline comparison, evaluates the model, and calculates a cost matrix.
9. **Model Explanation:** Visualizes the Decision Tree, plots feature importances for both Decision Tree and Random Forest models, and provides local explanations for individual predictions.
10. **Export Data:** Exports the preprocessed training and testing data to CSV files.

To run the script, simply execute it in your Python environment:
```sh
python main.py
```

## Features
- Data exploration and visualization
- Preprocessing with `ColumnTransformer` and pipelines
- Training and tuning Decision Tree and Random Forest classifiers using `GridSearchCV`
- Comparison with a Dummy Classifier
- Evaluation using accuracy, classification report, confusion matrix, and cost matrix
- Visualization of decision trees and feature importances
- Local explanation for individual predictions


## Contributors
- Jan Alexandr Kopřiva ([koprjaa](https://github.com/koprjaa))
- David Hložek
- Jakub Hermann
- Ondrej Čech
- Milan Tvrdík

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
