import pandas as pd


df = pd.read_csv('set A corporate_rating.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())