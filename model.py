import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Charger les données
data = pd.read_csv('MaskBeliefs.csv')

# Afficher les informations sur le dataset
print(data.info())

# Séparer les features (X) et la variable cible (y)
y = data['Public']
X = data.drop('Public', axis=1)

# Traiter les colonnes catégorielles
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Traiter la colonne 'Timestamp'
X['Timestamp'] = pd.to_datetime(X['Timestamp']).astype(int) / 10**9

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encoder la variable cible
le = LabelEncoder()
y = le.fit_transform(y)

# Afficher les noms des colonnes utilisées pour la prédiction
print("Colonnes utilisées pour la prédiction :", X.columns)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Afficher l'importance des features
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
print("\nImportance des features :")
print(feature_importance.sort_values('importance', ascending=False))