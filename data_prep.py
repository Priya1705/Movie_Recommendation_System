import numpy as numpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('ratings.csv')
data=data.iloc[:,0:3]
data.to_csv('ratings.csv' ,index=False)

data1=pd.read_csv('ratings.csv')
data2=pd.read_csv('movies.csv')
dataset=pd.merge(data1, data2, on='movieId')
dataset.to_csv('dataset.csv')

dataset=pd.read_csv('dataset.csv')
df = pd.DataFrame(dataset.title.str.split('(',1).tolist(), columns=['title1','year'])
df.year=df.year.str.strip(')')
dt = pd.DataFrame(df.title1.str.split(',',1).tolist(), columns=['title','second'])
dt['year']=df['year'].str[-4:]
dt['title']=dt['title'].str.strip()
dt.drop('second', axis=1, inplace=True)
# dt['movieId']=dataset['movieId']
dt.to_csv('dt.csv')

data3=pd.read_csv('dataset.csv')
data3.drop('title', axis=1, inplace=True)
data3.drop('genres', axis=1, inplace=True)
data4=pd.read_csv('dt.csv')
data_final=pd.merge(data3, data4, on='index')
data_final.drop('index', axis=1, inplace=True)
# # data_final['ratings']=data_final['rating']
# # data_final.drop('rating', axis=1, inplace=True)
# # data.final['title']=data_final['title'].strip()
data_final.to_csv('data_final.csv', index=False)
print(data_final.head())
