# %%
"""
# Template for coursework - Part 2: Clustering


"""

# %%
# import of all libraries
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics, set_config
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from yellowbrick.cluster import KElbowVisualizer

# %%
"""
## Team identification
"""

# %%
r"""
- 4IZ110 Wednesday 12:45\-13:30
- Team A
- David Hložek, Jan Alexandr Kopřiva, Jakub Hermann, Ondrej Čech, Milan Tvrdík


"""

# %%
"""
# Introduction
"""

# %%
r"""
1. This term paper deals with the detection of cardiovascular diseases \(heart diseases\) using machine learning in the Python language. This issue has considerable potential for improving health care and reducing costs in the health sector. Timely and accurate diagnosis of heart diseases with the help of allows adequate treatment to be started earlier, which can save lives and reduce the risk of serious complications. At the same time, early identification of patients at risk offers scope for effective preventive measures. Our group believes that the application of machine learning methods in the field of cardiovascular medicine could bring benefits to improve the quality of life of patients and save human lives.
2. Link to the dataset: [https://www.kaggle.com/datasets/fedesoriano/heart\-failure\-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)


"""

# %%
"""
## Customization
"""

# %%
r"""
1. Target Attribute: HeartDisease
2. Instance of interest: 69th row \(see below\)
3. Attribute of interest: Cholesterol


"""

# %%
# Import dataset
heart_data = pd.read_csv('heart.csv')

# Show instance nr. 69
chosen_instance = heart_data.iloc[68]
print(chosen_instance, "\n")

# %%
# Cost matrix
tp = 0
fn = 10 # False negatives have the highest cost
tn = 0
fp = 1 # False positive is not that important

cost_matrix = np.array([[tn, fp],
                        [fn, tp]])

# Output the cost matrix
print(cost_matrix)

# %%
"""
# Data preprocessing
"""

# %%
"""
## Preprocessing for unsupervised machine learning
"""

# %%
"""
* As clustering is  performed only on the chosen subset of data, remove data not in the subgroup
* Perform min-max feature rescaling
* remove target attribute from clustering
* do *not* create train-test splits
"""

# %%
# Load the heart dataset from CSV file # No rows are excluded, because it decreases accuracy of the models
data = pd.read_csv('heart.csv')

# Display the first 5 rows of the dataset
data.head(5)

# %%
# Separate features (train) and target variable (test)
train, test = data.drop('HeartDisease', axis=1), data[['HeartDisease']]

# %%
# Scale numerical features and encode non-numerical features
set_config(transform_output="pandas")

# Define numerical and categorical features
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Define transformers for numerical and categorical features
numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

# Preprocess data with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)],
    verbose_feature_names_out=False
)

# Fit preprocessing on training data
preprocessor.fit(train)

# Transform dataset for training and evaluation
train_preprocessed = preprocessor.transform(train)
test_preprocessed = pd.concat([train_preprocessed, data['HeartDisease']], axis=1)

# %%
# Display the first 5 rows of the preprocessed training data
train_preprocessed.head(5)

# %%
# Display the first 5 rows of the preprocessed test data
test_preprocessed.head(5)

# %%
"""
## Modeling and visualization
"""

# %%
"""
### Clustering 1 (k-means)


"""

# %%
"""
* Use the elbow curve graph to find the best value of inertia (wcss)
* It is recommended to use two measures (such as the Silhouette score and the Inertia) and compare the resulting number of clusters based on their best values.
* Create a scatter plot for each cluster and use color
"""

# %%
# Initialize and fit KMeans clustering model with 2 clusters
k2_model = KMeans(n_clusters=2, random_state=42, n_init=10)
k2_model.fit(train_preprocessed)

# %%
# Initialize KMeans model for tuning
kmeans_tune = KMeans(random_state=42, n_init=10)

# Inertia method to find best k value (best-k = 6)
KElbowVisualizer(kmeans_tune, k=(2, 15), metric="distortion").fit(train_preprocessed).show()

# Silhouette method to find best k value (best-k = 2)
KElbowVisualizer(kmeans_tune, k=(2, 15), metric="silhouette").fit(train_preprocessed).show()


# %%
# Initialize KMeans model with 6 clusters as determined by the best k value based on inertia
k_best_model = KMeans(n_clusters=6, random_state=42)

# Fit the KMeans model to the preprocessed training data
k_best_model.fit(train_preprocessed)

# %%
def plot_clusters(data, clusters):
    """
    Plot the clusters formed by KMeans algorithm.

    Parameters:
    - data: Preprocessed data used for clustering
    - clusters: Fitted KMeans clustering model
    """
    for cluster_num in range(clusters.n_clusters):
        cluster_mask = (clusters.labels_ == cluster_num)
        cluster_data = data[cluster_mask].values
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster_num}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-Means Clusters (PCA-reduced data)")
    plt.legend()
    plt.show()

# %%
# Perform PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2, random_state=42)

# Fit PCA and transform preprocessed training data
idk = pca.fit_transform(train_preprocessed)

# Plot clusters formed by KMeans algorithm on PCA-reduced data
plot_clusters(idk, k_best_model)

# %%
"""
### Clustering  2 (hierarchical)
"""

# %%
"""
* Try to use the dendrogram to identify outliers. If an instance joins a cluster higher on the dendrogram, it generally means it is less similar to the other instances.
"""

# %%
# Perform hierarchical clustering and generate dendrogram visualization
linked = linkage(train_preprocessed, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()

# %%
"""
# Evaluation
"""

# %%
"""
## Global
"""

# %%
"""
### Clustering 1 (k-means)
"""

# %%
"""
* Compute the rand index using the value of the target attribute
"""

# %%
k2_clusters = list(k2_model.predict(train_preprocessed))
k_best_clusters = list(k_best_model.predict(train_preprocessed))
random_clusters = [random.randint(0, 1) for i in range(train_preprocessed.shape[0])]

real_clusters = list(test["HeartDisease"])


print("K2 result: ", metrics.rand_score(k2_clusters, real_clusters))
print("K-best result: ", metrics.rand_score(k_best_clusters, real_clusters))
print("Random result: ", metrics.rand_score(random_clusters, real_clusters))

# %%
"""
### Clustering 2 (hierarchical)
"""

# %%
"""
* Compute the rand index using the value of the target attribute


"""

# %%
# Initialize Agglomerative Clustering model with 6 clusters
hierarchical_model = AgglomerativeClustering(n_clusters=6)

# Predict cluster labels using hierarchical clustering model
hierarchical_clusters = list(hierarchical_model.fit_predict(train_preprocessed))

# Calculate Rand index score between predicted hierarchical clusters and real clusters
print("Hierarchical clustering result (Rand score): ", metrics.rand_score(hierarchical_clusters, real_clusters))

# %%
"""
## Local
"""

# %%
"""
*	Use the model to classify the chosen instance into a cluster
"""

# %%
# selected instance number 69 with heart failure
chosen_instance = heart_data.iloc[68]
print(chosen_instance, "\n")

# instance 69 after applying preprocessing
chosen_instance_preprocessed = preprocessor.transform(train).iloc[68:69]

# prediction
k2_prediction = k2_model.predict(chosen_instance_preprocessed)
k_best_prediction = k_best_model.predict(chosen_instance_preprocessed)

print("Cluster according to k2: ", k2_prediction[0])
print("Cluster according to k_best: ", k_best_prediction[0], "\n")

# %%
"""
# Explanation
"""

# %%
"""
## Global explanation
"""

# %%
r"""
### Clustering 1 \- KMeans


"""

# %%
def agg_func(x):
    """
    Aggregate function to apply to grouped data.

    Parameters:
    - x: Grouped data for each feature

    Returns:
    - Aggregated value (most frequent value for object data types, mean for numerical data types)
    """
    if heart_data.dtypes[x.name] == "object":
        # For nominal values, return the most frequent value
        return x.value_counts().index[0]
    # For numerical values, return the mean
    return x.mean()

# %%
# Concatenate original data with predicted cluster labels from KMeans model
data_plus_cluster = pd.concat([heart_data, pd.Series(k2_model.predict(train_preprocessed), name="Cluster")], axis=1)

# Group data by cluster and apply aggregation functions
data_plus_cluster.groupby(["Cluster"]).agg(agg_func)

# %%
# Concatenate original data with predicted cluster labels from the best KMeans model
data_plus_cluster = pd.concat([heart_data, pd.Series(k_best_model.predict(train_preprocessed), name="Cluster")], axis=1)

# Group data by cluster and apply aggregation functions using agg_func
data_plus_cluster.groupby(["Cluster"]).agg(agg_func)

# %%
"""
*	Interpret the final clusters based on their centroids and the number of instances in each cluster.
"""

# %%
"""
## Interpretation


"""

# %%
"""

"""

# %%
"""
### Clustering 2 - hierarchical
"""

# %%
# Perform hierarchical clustering and generate dendrogram using Ward linkage method
linked = linkage(train_preprocessed, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()
