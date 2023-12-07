import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=13)

mlp = MLPClassifier(random_state=0)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

plt.plot(mlp.loss_curve_)
plt.title("Courbe d'apprentissage")
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.show()

y_test_pred = mlp.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_test_pred)
rand_index = metrics.adjusted_rand_score(y_test, y_test_pred)

total_fraudes_detectees = sum(y_test_pred)

print(f'Accuracy : {accuracy*100:.4f}%')
print(f'Indice de Rand : {rand_index:.4f}')
print(f'Nombre total de fraudes détectées : {total_fraudes_detectees}')
