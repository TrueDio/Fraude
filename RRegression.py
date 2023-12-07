import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Erreur quadratique: {mse_ridge:.4f}")

y_pred_ridge_binary = (y_pred_ridge > 0.5).astype(int)
accuracy_ridge = accuracy_score(y_test, y_pred_ridge_binary)
print(f'Accuracy: {accuracy_ridge * 100:.4f}%')

fraudes_detectees = sum(y_pred_ridge_binary)
print(f"Nombre de fraudes détectées : {fraudes_detectees}")
