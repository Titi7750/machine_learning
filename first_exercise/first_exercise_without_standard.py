import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x = np.array([
    [170, 0], # 0 = Football, 1 = Basketball
    [185, 1],
    [160, 0],
    [200, 1],
    [175, 0]
])

y = np.array(["Amateur", "Pro", "Amateur", "Pro", "Amateur"])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)

new_player = np.array([[180, 1]])
prediction = knn.predict(new_player)

print(f'Joueur prédit : {prediction[0]}')
