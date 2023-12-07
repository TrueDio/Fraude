import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import silhouette_score

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=13)

def plot_results(X, y_pred, title):
    if X.shape[1] == 2:
        unique_clusters = list(set(y_pred))
        for cluster in unique_clusters:
            cluster_points = X[y_pred == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=50)
        plt.title(title)
        plt.legend()
        plt.show()
    elif X.shape[1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        unique_clusters = list(set(y_pred))
        for cluster in unique_clusters:
            cluster_points = X[y_pred == cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster}', s=50)
        ax.set_xlabel('amount')
        ax.set_ylabel('oldbalanceOrg')
        ax.set_zlabel('newbalanceOrig')
        ax.set_title(title)
        ax.set_xlim(X[:, 0].min(), X[:, 0].max())
        ax.set_ylim(X[:, 1].min(), X[:, 1].max())
        ax.set_zlim(X[:, 2].min(), X[:, 2].max())
        ax.legend()
        plt.show()

def evaluate_performance(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    rand_index = metrics.adjusted_rand_score(y_true, y_pred)
    silhouette = silhouette_score(X_scaled, y_pred)
    print(f'Accuracy: {accuracy*100:.4f}%')
    print(f'Indice de rand: {rand_index*(-1):.4f}')
    print(f'Silhouette: {silhouette:.4f}')

    total_fraudes = sum((y_true == 1) & (y_pred == cluster) for cluster in set(y_pred))
    print(f'Total de fraudes : {total_fraudes}')

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

evaluate_performance(y, y_kmeans)
plot_results(X_scaled, y_kmeans, "K-Means")

