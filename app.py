import pickle
import os
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Chemin vers les modèles
model_path = os.path.join(os.path.dirname(__file__), "models")

# Chargement des modèles
models = {
    "KNN": pickle.load(open(os.path.join(model_path, 'KNN_model.pkl'), 'rb')),
    "Decision Tree": pickle.load(open(os.path.join(model_path, 'Decision Tree_model.pkl'), 'rb')),
    "Naive Bayes": pickle.load(open(os.path.join(model_path, 'Naive Bayes_model.pkl'), 'rb'))
}

# Charger le dataset original pour configurer l'encodage
df=pd.read_csv("C:\\Users\\Aziz\\Downloads\\ObesityDataSet.csv")

# Encodage des colonnes catégorielles
label_encoders = {}
cat_cols = df.select_dtypes(include="object").drop(columns=["NObeyesdad"]).columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encodage de la variable cible
target_encoder = LabelEncoder()
df["NObeyesdad"] = target_encoder.fit_transform(df["NObeyesdad"])

# Standardisation
scaler = StandardScaler()
X = df.drop("NObeyesdad", axis=1)
scaler.fit(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupération des données utilisateur
    user_data = {
        'Gender': request.form['Gender'],
        'Age': float(request.form['age']),
        'Height': float(request.form['height']),
        'Weight': float(request.form['weight']),
        'family_history_with_overweight': request.form['family_history_with_overweight'],
        'FAVC': request.form['FAVC'],
        'FCVC': float(request.form['FCVC']),
        'NCP': float(request.form['NCP']),
        'CAEC': request.form['CAEC'],
        'SMOKE': request.form['SMOKE'],
        'CH2O': float(request.form['CH2O']),
        'SCC': request.form['SCC'],
        'FAF': float(request.form['FAF']),
        'TUE': float(request.form['TUE']),
        'CALC': request.form['CALC'],
        'MTRANS': request.form['MTRANS']
    }

    # Transformer en DataFrame
    user_df = pd.DataFrame([user_data])

    # Encodage des colonnes catégorielles
    for col in cat_cols:
        le = label_encoders[col]
        user_df[col] = le.transform(user_df[col])

    # Mise à l'échelle
    user_df_scaled = scaler.transform(user_df)

    # Sélection du modèle
    selected_model = request.form['model']
    model = models[selected_model]

    # Prédiction
    prediction = model.predict(user_df_scaled)
    label = target_encoder.inverse_transform([prediction[0]])[0]

    return render_template('result.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
