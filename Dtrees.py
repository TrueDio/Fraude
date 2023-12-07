import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=50, random_state=42)

dt_model.fit(X_train, y_train)

predictions = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'Accuracy: {accuracy * 100:.4f}%')
print(f'Erreur quadratique: {mse:.4f}')

print('Classement:')
print(classification_report(y_test, predictions))

nombre_fraudes_detectees = sum(predictions)
print(f'Nombre de fraudes détectées : {nombre_fraudes_detectees}')
