import pandas as pd
import plotly.express as px

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
