import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=13)

gmm = GaussianMixture(n_components=3, random_state=42)
y_gmm = gmm.fit_predict(X_scaled)

data['cluster'] = y_gmm

total_frauds = 0

for cluster in data['cluster'].unique():
    cluster_points = data[data['cluster'] == cluster]
    frauds_in_cluster = cluster_points['isFraud'].sum()
    total_frauds += frauds_in_cluster
    total_points_in_cluster = len(cluster_points)

    print(f"Cluster {cluster}: {frauds_in_cluster} fraudes sur {total_points_in_cluster} points")

print(f"\nTotal de fraudes dans l'ensemble du dataset : {total_frauds}")

def evaluate_performance(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    rand_index = metrics.adjusted_rand_score(y_true, y_pred)
    print(f'\nAccuracy: {accuracy*100:.4f}%')
    print(f'Indice de rand: {rand_index:.4f}')

y_test_pred = gmm.predict(X_test)
evaluate_performance(y_test, y_test_pred)