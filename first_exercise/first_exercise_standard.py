import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

x = np.array([
    [170, 0], # 0 = Football, 1 = Basketball
    [185, 1],
    [160, 0],
    [200, 1],
    [175, 0]
])

y = np.array(["Amateur", "Pro", "Amateur", "Pro", "Amateur"])

preprocessor = ColumnTransformer(
    transformers=[
        ("size", StandardScaler(), [0])
    ],
    remainder='passthrough'
)

knn = KNeighborsClassifier(n_neighbors=3)
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', knn)])
pipe.fit(x, y)

new_player = np.array([[180, 1]])
prediction = pipe.predict(new_player)

print(f'Joueur prédit : {prediction[0]}')
