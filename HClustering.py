import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def plot_results_3d(X, y_pred, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster in range(len(set(y_pred))):
        cluster_points = X[y_pred == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   label=f'Cluster {cluster}', s=50)

    ax.set_xlabel('amount')
    ax.set_ylabel('oldbalanceOrg')
    ax.set_zlabel('newbalanceOrig')
    ax.set_title(title)
    ax.legend()

    plt.show()

def evaluate_performance(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    rand_index = metrics.adjusted_rand_score(y_true, y_pred)
    silhouette = metrics.silhouette_score(X_scaled, y_pred)
    
    print(f'Accuracy: {accuracy*100:.4f}%')
    print(f'Indice rand: {rand_index*(-1):.4f}')
    print(f'Silhouette Score: {silhouette:.4f}')

    return accuracy, rand_index, silhouette

agg_clustering = AgglomerativeClustering(n_clusters=3)
y_agg = agg_clustering.fit_predict(X_scaled)

plot_results_3d(X_scaled, y_agg, "Hierarchical Clustering")
accuracy, rand_index, silhouette = evaluate_performance(y, y_agg)

total_fraudes = sum(y_agg == 1)
print(f"Nombre total de fraudes trouvées : {total_fraudes}")

