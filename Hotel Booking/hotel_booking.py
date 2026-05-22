import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold      import TSNE
from sklearn.cluster       import KMeans
from sklearn.metrics       import silhouette_score, davies_bouldin_score
print('All imports OK')
hb_df = pd.read_csv(r'C:\Users\parth\Desktop\week-3\Hotel Booking\hotel_bookings.csv')
hb_df.head()
hb_df.info()
hb_df.isnull().sum()
hb_df.duplicated().sum()
hb_df.drop_duplicates(inplace=True)
hb_df.duplicated().sum()
hb_df.isnull().sum()
hb_df = hb_df.drop(columns=['country', 'agent', 'company','children'])
hb_df.isnull().sum()
non_numeric_cols = hb_df.select_dtypes(exclude='number').columns
hb_df = hb_df.drop(columns=non_numeric_cols)
X = hb_df
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
corr = hb_df.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Correlation')

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)

# Annotate each cell
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                fontsize=7, color='black' if abs(corr.iloc[i,j]) < 0.7 else 'white')

ax.set_title('Feature Correlation Matrix — Cancer Dataset', fontweight='bold', pad=14)
plt.tight_layout()
plt.show()
experiment_name = "Parth_d"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
else: 
    mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    scores = []

    K = range(2, 11)

    for k in K:

        model = KMeans(
            n_clusters=k,
            init='k-means++',
            random_state=42
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        scores.append(score)

    # Plot
    plt.figure(figsize=(8,5))

    plt.plot(K, scores, marker='o')

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.show()
    
    #------------
    scores = []

    K = range(2, 11)

    for k in K:

        model = KMeans(
            n_clusters=k,
            init='k-means++',
            random_state=42
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        scores.append(score)

    # Plot
    plt.figure(figsize=(8,5))

    plt.plot(K, scores, marker='o')

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.show()
    #-----
    best_k = 3

    model = KMeans(
        n_clusters=best_k,
        init='k-means++',
        random_state=42
    )

    clusters = model.fit_predict(X_scaled)

    hb_df['cluster'] = clusters
    #-----
    cluster_summary = hb_df.groupby('cluster').mean()
    print(cluster_summary)
    #------
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))

    plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=hb_df['cluster']
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Hotel Clusters')
    plt.show()
    
    mlflow.sklearn.log_model(model, "model")
    print("run_id:", run.info.run_id)
    print("model_uri:", f"runs:/{run.info.run_id}/model")
#--------------------------------------------------------------------



import os
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

print("All imports OK")

# =========================================================
# LOAD DATA
# =========================================================

hb_df = pd.read_csv(
    r'C:\Users\parth\Desktop\week-3\Hotel Booking\hotel_bookings.csv'
)

print(hb_df.head())

# =========================================================
# DATA CLEANING
# =========================================================

print("Duplicates:", hb_df.duplicated().sum())

hb_df.drop_duplicates(inplace=True)

# Drop unwanted columns
hb_df = hb_df.drop(
    columns=['country', 'agent', 'company', 'children'],
    errors='ignore'
)

# Keep only numeric columns
non_numeric_cols = hb_df.select_dtypes(exclude='number').columns
hb_df = hb_df.drop(columns=non_numeric_cols)

print("Final Shape:", hb_df.shape)

# =========================================================
# FEATURE SCALING
# =========================================================

X = hb_df.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# MLFLOW SETUP
# =========================================================

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

experiment_name = "Parth_d"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
else: 
    mlflow.set_experiment(experiment_name)

# =========================================================
# START RUN
# =========================================================

with mlflow.start_run() as run:

    # -----------------------------------------------------
    # LOG BASIC INFO
    # -----------------------------------------------------

    mlflow.log_param("dataset", "hotel_bookings.csv")
    mlflow.log_param("algorithm", "KMeans")
    mlflow.log_param("scaler", "StandardScaler")

    # =====================================================
    # CORRELATION MATRIX
    # =====================================================

    corr = hb_df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(
        corr,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        aspect='auto'
    )

    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha='right',
        fontsize=8
    )

    ax.set_yticklabels(
        corr.columns,
        fontsize=8
    )

    ax.set_title("Correlation Matrix")

    plt.tight_layout()

    corr_plot = "correlation_matrix.png"
    plt.savefig(corr_plot)

    mlflow.log_artifact(corr_plot)

    plt.close()

    # =====================================================
    # SILHOUETTE ANALYSIS
    # =====================================================

    scores = []

    K = range(2, 11)

    for k in K:

        model = KMeans(
            n_clusters=k,
            init='k-means++',
            random_state=42
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        scores.append(score)

    # Best K
    best_k = K[np.argmax(scores)]

    # Log best K
    mlflow.log_param("best_k", best_k)

    # -----------------------------------------------------
    # SILHOUETTE PLOT
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(K, scores, marker='o')

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")

    silhouette_plot = "silhouette_analysis.png"

    plt.savefig(silhouette_plot)

    mlflow.log_artifact(silhouette_plot)

    plt.close()

    # =====================================================
    # FINAL MODEL
    # =====================================================

    final_model = KMeans(
        n_clusters=best_k,
        init='k-means++',
        random_state=42
    )

    clusters = final_model.fit_predict(X_scaled)

    hb_df['cluster'] = clusters

    # =====================================================
    # METRICS
    # =====================================================

    silhouette = silhouette_score(X_scaled, clusters)

    db_score = davies_bouldin_score(X_scaled, clusters)

    mlflow.log_metric("silhouette_score", silhouette)

    mlflow.log_metric("davies_bouldin_score", db_score)

    print("Silhouette Score:", silhouette)
    print("Davies Bouldin Score:", db_score)

    # =====================================================
    # CLUSTER SUMMARY
    # =====================================================

    cluster_summary = hb_df.groupby('cluster').mean()

    print(cluster_summary)

    cluster_summary.to_csv("cluster_summary.csv")

    mlflow.log_artifact("cluster_summary.csv")

    # =====================================================
    # PCA VISUALIZATION
    # =====================================================

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=hb_df['cluster']
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Hotel Clusters")

    pca_plot = "pca_clusters.png"

    plt.savefig(pca_plot)

    mlflow.log_artifact(pca_plot)

    plt.close()

    # =====================================================
    # LOG MODEL
    # =====================================================

    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model"
    )

    print("Run ID:", run.info.run_id)

    print(
        "Model URI:",
        f"runs:/{run.info.run_id}/model"
    )