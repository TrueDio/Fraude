import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = xgb.XGBClassifier()

evals = [(X_test, y_test)]
clf.fit(X_train, y_train, eval_set=evals, verbose=True)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'Accuracy: {accuracy * 100:.4f}%')
print(f'Erreur quadratique: {mse:.4f}')

print('Classement:')
print(classification_report(y_test, predictions))

fraudes_detectees = sum(predictions)
print(f'Nombre de fraudes détectées : {fraudes_detectees}')
