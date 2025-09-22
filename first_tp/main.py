import os
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create output directory if it doesn't exist
if not os.path.exists('./html'):
    os.makedirs('./html')

# 1. Read and clean the data
dataframe = pd.read_csv('./data/paris_airbnb.csv')

print(f'Dataframe shape: {dataframe.shape}')
print(f'Header of the dataframe:\n{dataframe.head()}')
print(f'Info of the dataframe: {dataframe.info()}')
print(f'Description of the dataframe:\n{dataframe.describe()}')

columns_to_keep = [
    'price',
    'accommodates',
    'bedrooms',
    'latitude',
    'longitude',
    'minimum_nights',
    'number_of_reviews',
    'room_type'
]

dataframe = dataframe[columns_to_keep]

print(f'New dataframe shape: {dataframe.shape}')
print(f'New header of the dataframe:\n{dataframe.head()}')
print(f'New info of the dataframe: {dataframe.info()}')
print(f'New description of the dataframe:\n{dataframe.describe()}')

dataframe['price'] = dataframe['price'].replace('[\$,€,\s]', '', regex=True).astype(float)

dataframe = dataframe.dropna(subset=['price'])
dataframe = dataframe[dataframe['price'] > 0]

with open('./data/cleaned_paris_airbnb.csv', 'w') as cleaned_file:
    dataframe.to_csv(cleaned_file, index=False)


dataframe = pd.read_csv('./data/cleaned_paris_airbnb.csv')

# 2. Generate and save figures with plotly
scatter_plot = px.scatter(
    data_frame=dataframe,
    x='accommodates',
    y='price',
    color='room_type'
)
scatter_plot.write_html('./html/scatter_plot.html')

histogram = px.histogram(
    data_frame=dataframe,
    x='price',
    y='room_type'
)
histogram.write_html('./html/histogram.html')

box_plot = px.box(
    data_frame=dataframe,
    x='price',
    y='room_type'
)
box_plot.write_html('./html/box_plot.html')

# 3. Split / Standardize / Pipeline
y = dataframe['price']
X = dataframe.drop(columns=['price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

nums_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, nums_cols),
        ('cat', categorical_pipeline, cat_cols)
    ],
    remainder='passthrough'
)

knn = KNeighborsClassifier(n_neighbors=5)
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', knn)])
pipe.fit(X_train, y_train)

# 4. KNN Regression & Evaluation
prediction = pipe.predict(X_test)

mae = mean_absolute_error(y_test, prediction)
rmse = mean_squared_error(y_test, prediction)

print(f'Mean Absolute Error (MAE): {round(mae, 2)}€')
print(f'Root Mean Squared Error (RMSE): {round(rmse, 2)}€')
