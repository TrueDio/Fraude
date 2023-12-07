import pandas as pd

def nettoyer_dataset(file_path, output_file='cdata.csv'):
    data = pd.read_csv(file_path, sep=';')
    data = data.fillna(data.select_dtypes(include='number').mean())
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)

    if 'isFlaggedFraud' in data.columns:
        data = data.drop('isFlaggedFraud', axis=1)
    
    if 'type' in data.columns:
        data = data.drop('type', axis=1)

    data.to_csv(output_file, index=False, sep=',')
    print(f"\nLe jeu de données nettoyé a été sauvegardé dans '{output_file}'.")

path = "C:/Users/dimit/OneDrive/Bureau/Guardia/Projet cybersécurité/Analyse de fraude financière/data.csv"
output_filename = "C:/Users/dimit/OneDrive/Bureau/Guardia/Projet cybersécurité/Analyse de fraude financière/cdata.csv"
nettoyer_dataset(path, output_filename)