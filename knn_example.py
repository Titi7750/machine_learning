import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1 . Préparation des données
x = np.array([
    [150, 1],
    [180, 1],
    [120, 0]
])

y = np.array(["Pomme", "Pomme", "Orange"])

# 2. Standardisation des données (poids)
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('weight', StandardScaler(), [0]),  # Standardiser la colonne 0 (poids)
#         ("color", "passthrough", [1])  # Ne pas transformer la colonne 1 (couleur)
#     ]
# )

# OR

preprocessor = ColumnTransformer(
    transformers=[
        ('weight', StandardScaler(), [0])
    ],
    remainder='passthrough'
)

# 3. Création et entraînement du modèle
knn = KNeighborsClassifier(n_neighbors=1)
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', knn)])
pipe.fit(x, y)

# 4. Prédiction
new_fruit = np.array([[130, 1]])
prediction = pipe.predict(new_fruit)

print(prediction[0])