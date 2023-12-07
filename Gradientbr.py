import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

binary_predictions = np.round(predictions)

mse = mean_squared_error(y_test, predictions)
print(f'Erreur quadratique : {mse:.4f}')

accuracy = accuracy_score(y_test, binary_predictions)
print(f'Accuracy: {accuracy * 100:.4f}%')

print('Classement:')
print(classification_report(y_test, binary_predictions))

nombre_fraudes_detectees = sum(binary_predictions)
print(f'Nombre de fraudes détectées : {int(nombre_fraudes_detectees)}')
