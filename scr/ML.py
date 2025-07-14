import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import numpy as np

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"Matrice de confusion - {model.__class__.__name__}")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Charger les données
df = pd.read_csv("C:\\Users\\Aziz\\Downloads\\ObesityDataSet.csv")
print("🔍 Premières lignes :\n", df.head())
print("\n📊 Infos :\n", df.info())
print("\n❓ Valeurs manquantes :\n", df.isnull().sum())
print("\n📈 Statistiques :\n", df.describe(include='all'))

# Encodage
label_encoders = {}
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Séparation X/y
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("✅ Données prêtes !")

# Modèles
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Résultats
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n🔹 {name} 🔹")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

    plot_confusion_matrix(model, X_test, y_test)

    # Sauvegarder modèle
    with open(f"models/{name}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Modèle {name} sauvegardé.")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    })

# Résumé DataFrame
df_results = pd.DataFrame(results)
print("\n📋 Résumé des performances :\n", df_results)

# ---------- VISUALISATIONS ---------- #

# 🔹 1. Barplot groupé : Accuracy, Precision, Recall, F1-score
df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Set2")
plt.title("Comparaison des métriques par modèle")
plt.ylim(0, 1.05)
plt.legend(title="Métrique")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 🔹 2. Heatmap des scores
plt.figure(figsize=(8, 4))
sns.heatmap(df_results.set_index("Model"), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap des scores des modèles")
plt.tight_layout()
plt.show()

# 🔹 3. Radar chart
labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # boucle

plt.figure(figsize=(8, 8))
for i, row in df_results.iterrows():
    values = row[labels].tolist()
    values += values[:1]
    plt.polar(angles, values, label=row["Model"], linewidth=2)
    plt.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], labels)
plt.title("Radar Chart - Performance des modèles")
plt.ylim(0, 1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# 🔹 4. Barplot simple : Accuracy uniquement
plt.figure(figsize=(8, 6))
sns.barplot(data=df_results, x="Model", y="Accuracy", palette="pastel")
plt.title("Comparaison des Accuracy des modèles")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.show()
