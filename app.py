import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Fonction pour charger et prétraiter les données
@st.cache_data
def load_data():
    data = pd.read_csv('MaskBeliefs.csv')
    return data

# Charger les données
data = load_data()

# Prétraitement des données
def preprocess_data(data):
    # Séparer les features (X) et la variable cible (y)
    y = data['Public']
    X = data.drop('Public', axis=1)

    # Identifier les colonnes catégorielles (excluant 'Timestamp')
    categorical_columns = X.select_dtypes(include=['object']).columns.drop('Timestamp')

    # Traiter les colonnes catégorielles
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Traiter la colonne 'Timestamp'
    X['Timestamp'] = pd.to_datetime(X['Timestamp']).astype(int) / 10**9

    # Encoder la variable cible
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, categorical_columns

X, y, categorical_columns = preprocess_data(data)

# Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

st.title('Prédiction des croyances sur les masques')

# Créer des inputs pour chaque feature
features = {}
for column in X.columns:
    if column == 'Timestamp':
        features[column] = st.date_input(f"Sélectionnez la date pour {column}", pd.to_datetime('2020-09-25'))
    elif X[column].dtype == 'int64':
        features[column] = st.number_input(f"Sélectionnez la valeur pour {column}", int(X[column].min()), int(X[column].max()), int(X[column].mean()))
    else:
        features[column] = st.selectbox(f"Sélectionnez la valeur pour {column}", sorted(X[column].unique()))

# Créer un bouton pour faire la prédiction
# if st.button('Prédire'):
#     input_df = pd.DataFrame([features])
    
#     # Prétraiter les données d'entrée
#     for col in input_df.columns:
#         if col in categorical_columns:
#             le = LabelEncoder()
#             le.fit(X[col])
#             input_df[col] = le.transform(input_df[col].astype(str))
    
#     # Convertir la date en timestamp
#     input_df['Timestamp'] = pd.to_datetime(input_df['Timestamp']).astype(int) / 10**9
    
#     input_scaled = scaler.transform(input_df)
#     prediction = model.predict(input_scaled)
    
#     st.write(f"La prédiction pour 'Public' est : {'Oui' if prediction[0] == 1 else 'Non'}")


# Créer un bouton pour faire la prédiction
if st.button('Prédire'):
    input_df = pd.DataFrame([features])
    
    # Prétraiter les données d'entrée
    for col in input_df.columns:
        if col in categorical_columns:
            le = LabelEncoder()
            le.fit(X[col])
            input_df[col] = le.transform(input_df[col].astype(str))
    
    # Convertir la date en timestamp
    input_df['Timestamp'] = pd.to_datetime(input_df['Timestamp']).astype(int) / 10**9
    
    input_scaled = scaler.transform(input_df)
    
    # Utiliser predict_proba au lieu de predict
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Afficher les probabilités pour chaque classe
    st.write(f"Probabilité que 'Public' soit Non : {prediction_proba[0]:.2%}")
    st.write(f"Probabilité que 'Public' soit Oui : {prediction_proba[1]:.2%}")
    
    # Afficher la prédiction finale
    prediction = "Oui" if prediction_proba[1] > 0.5 else "Non"
    st.write(f"Prédiction finale pour 'Public' : {prediction}")

# Afficher quelques statistiques sur le dataset
st.subheader('Statistiques du dataset')
st.write(data.describe())