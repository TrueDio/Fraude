import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)

y_pred_logistic = logistic_reg.predict(X_test)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"Accuracy: {accuracy_logistic*100:.4f}%")
print("Classement:\n", classification_report(y_test, y_pred_logistic))

mse_logistic = mean_squared_error(y_test, y_pred_logistic)
print(f"Erreur quadratique: {mse_logistic:.4f}")

fraudes_detectees = sum(y_pred_logistic)
print(f"Nombre de fraudes détectées : {fraudes_detectees}")
