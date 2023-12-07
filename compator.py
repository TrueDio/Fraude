import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\data\data.csv"
df_dataset = pd.read_csv(path)

if "isFraud" in df_dataset.columns:
    pourcentage_fraude_reel = (df_dataset["isFraud"].sum() / len(df_dataset)) * 100
    print(f"Le pourcentage de fraude réel est de {pourcentage_fraude_reel}%.")
else:
    print("La colonne 'isFraud' n'existe pas dans l'ensemble de données.")

total_fraudes_reelles = df_dataset['isFraud'].sum()

dataset_df = pd.read_csv(path)

pourcentage_fraude = (dataset_df['isFraud'].sum() / len(dataset_df)) * 100

fichier_resultats = r"C:\Users\dimit\OneDrive\Bureau\Guardia\Projet cybersécurité\Analyse de fraude financière\result.xlsx"
results_df = pd.read_excel(fichier_resultats).sort_values(by='Nom')

results_df['Accuracy'] = results_df['Accuracy'].str.rstrip('%').astype('float') / 100.0
results_df['Erreur quadratique'] = results_df['Erreur quadratique'].replace('None', pd.NA).astype('float')
results_df['Indice de rand'] = results_df['Indice de rand'].replace('None', pd.NA).astype('float')

results_df['Fraude'] = pd.to_numeric(results_df['Fraude'], errors='coerce')

results_df = results_df.dropna(subset=['Fraude'])

clustering_results = results_df[results_df['Type'] == 'Clustering']
regression_results = results_df[results_df['Type'] == 'Regression']
tree_results = results_df[results_df['Type'] == 'Tree'].sort_values(by='Accuracy')

tree_results = tree_results.sort_values(by='Accuracy')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comparaison des Performances des Algorithmes')

axes[0, 0].scatter(clustering_results['Nom'], clustering_results['Indice de rand'], label='Indice de Rand', marker='o', color='orange')
axes[0, 0].plot(clustering_results['Nom'], clustering_results['Indice de rand'], linestyle='-', color='orange', linewidth=2, markersize=8)
axes[0, 0].set_ylabel('Indice de Rand')

axes[0, 1].scatter(regression_results['Nom'], regression_results['Erreur quadratique'], label='Mean Squared Error', marker='o', color='green')
axes[0, 1].plot(regression_results['Nom'], regression_results['Erreur quadratique'], linestyle='-', color='green', linewidth=2, markersize=8)
axes[0, 1].set_ylabel('Mean Squared Error')

axes[1, 0].scatter(tree_results['Nom'], tree_results['Accuracy'], label='Accuracy', marker='o', color='blue')
axes[1, 0].plot(tree_results['Nom'], tree_results['Accuracy'], linestyle='-', color='blue', linewidth=2, markersize=8)
axes[1, 0].set_ylabel('Accuracy')

axes[1, 1].scatter(results_df['Nom'], results_df['Fraude'], label='Fraudes Prédites', marker='o', color='purple')
axes[1, 1].plot(results_df['Nom'], results_df['Fraude'], linestyle='--', color='purple', linewidth=2, markersize=8)
axes[1, 1].axhline(y=total_fraudes_reelles, color='red', linestyle='--', label=f'Total Fraudes Réelles: {total_fraudes_reelles}')

for ax in axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Algorithmes')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

top_algo_proches = results_df.nlargest(13, 'Fraude', 'all').sort_values(by='Fraude', key=lambda x: abs(x - total_fraudes_reelles), ascending=True)

print(f'\nLe nombre total de fraudes réelles est {total_fraudes_reelles}. Voici les algorithmes les plus proches de cette valeur :\n')
print(top_algo_proches[['Nom', 'Fraude']])
