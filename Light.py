import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\cdata.csv"
data = pd.read_csv(path)

X = data.drop('isFraud', axis=1)
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

reg = lgb.LGBMClassifier()
reg.fit(X_train, y_train, eval_set=[(X_test, y_test)])

y_pred = reg.predict(X_test)

print('Classement:')
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.4f}%')

mse = mean_squared_error(y_test, y_pred)
print(f"Erreur quadratique : {mse:.4f}")

nombre_de_fraudes_trouvees = sum(y_pred)
print(f"Nombre de fraudes trouvées : {nombre_de_fraudes_trouvees}")
