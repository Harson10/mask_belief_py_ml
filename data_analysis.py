import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
data = pd.read_csv('MaskBeliefs.csv')  # Assurez-vous que le fichier CSV est dans le même dossier

# Afficher les premières lignes et les informations sur le dataset
print(data.head())
print(data.info())

# Visualiser la distribution de la variable cible (Public)
plt.figure(figsize=(10, 6))
sns.countplot(x='Public', data=data)
plt.title('Distribution de la variable Public')
plt.show()

# Analyser les corrélations entre les variables numériques
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()