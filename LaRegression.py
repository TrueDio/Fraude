import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Erreur quadratique: {mse_lasso:.4f}")

y_pred_lasso_binary = (y_pred_lasso > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_lasso_binary)
print(f'Accuracy: {accuracy * 100:.4f}%')

classification_rep = classification_report(y_test, y_pred_lasso_binary)
print("Classement:\n", classification_rep)

nombre_fraudes_predites = sum(y_pred_lasso_binary)
print(f"Nombre de fraudes prédites: {nombre_fraudes_predites}")
