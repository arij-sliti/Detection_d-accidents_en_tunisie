import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

DB_PATH = 'accidents.db'

conn = sqlite3.connect(DB_PATH)
query = """
SELECT v.id as video_id, v.name, v.accident_count, v.is_dangerous, v.cause, v.governorate, f.rating, f.comment, f.created_at
FROM videos v
JOIN feedback f ON v.id = f.video_id
"""
df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("Aucun feedback collecté pour l'entraînement.")
    exit()

# Afficher les commentaires
print(f"\n📝 {len(df)} retour(s) utilisateur :\n")
for idx, row in df.iterrows():
    print(f"Vidéo : {row['name']}")
    print(f"   Note : {row['rating']} étoile(s)")
    print(f"   Commentaire : {row['comment'] if row['comment'] else '(vide)'}")
    print(f"   Date : {row['created_at']}")
    print("-" * 50)

# Préparer les données pour le ML
df['satisfied'] = (df['rating'] >= 4).astype(int)

# Encodage
le_cause = LabelEncoder()
le_gov = LabelEncoder()
df['cause_enc'] = le_cause.fit_transform(df['cause'])
df['gov_enc'] = le_gov.fit_transform(df['governorate'])

features = ['accident_count', 'is_dangerous', 'cause_enc', 'gov_enc']
X = df[features]
y = df['satisfied']

# Si très peu de données, entraîner sans split
if len(df) < 5:
    print("\n⚠️ Peu de données, entraînement sur la totalité sans évaluation séparée.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    accuracy = None
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"\n📊 Précision du modèle sur l'ensemble de test : {accuracy:.2f}")

# Sauvegarde
joblib.dump(model, 'satisfaction_model.pkl')
joblib.dump(le_cause, 'le_cause.pkl')
joblib.dump(le_gov, 'le_gov.pkl')

print(f"\n✅ Modèle entraîné avec {len(df)} échantillons. Fichiers sauvegardés.")