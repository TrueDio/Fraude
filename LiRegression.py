import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)

y_pred_binary = (y_pred_linear >= 0.5).astype(int)

mse_linear = mean_squared_error(y_test, y_pred_linear)
accuracy = accuracy_score(y_test, y_pred_binary)

fraudes_detectees = sum(y_pred_binary)
print(f"Nombre de fraudes détectées : {fraudes_detectees}")

print(f"Erreur quadratique : {mse_linear:.4f}")
print(f"Accuracy : {accuracy*100:.4f}%")
