import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import matplotlib.pyplot as plt
# import seaborn as sns
import sys

m=sys.argv[1:]
y=""
r=len(m)
for i in range(r):
	y=y+" "+m[i]
print(y)
y=y.strip()

data=pd.read_csv('data_final.csv')

ratings=pd.DataFrame(data.groupby('title')['rating'].mean())
ratings['number_of_ratings']=data.groupby('title')['rating'].count()

# plt.hist(ratings['rating'] , bins=50)
# ratings['rating'].hist(bins=50)
# plt.hist(ratings['number_of_ratings'] , bins=50)
# plt.show()
# sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
# plt.show()

movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
# print(movie_matrix.head())

# print(ratings.sort_values('number_of_ratings', ascending=False).head(10))

# Shawshank_Redemption=movie_matrix['Shawshank Redemption']
# Silence_of_the_Lambs=movie_matrix['Silence of the Lambs']
# print(Silence_of_the_Lambs.head())
# print(Shawshank_Redemption.head())

# x='Silence of the Lambs'

x=movie_matrix[y]


# similar_to_Shawshank_Redemption=movie_matrix.corrwith(Shawshank_Redemption)
# print(similar_to_Shawshank_Redemption.head(20))

# similar_to_Silence_of_the_Lambs=movie_matrix.corrwith(Silence_of_the_Lambs)
# print(similar_to_Silence_of_the_Lambs.head(20))

similar_to_x=movie_matrix.corrwith(x)

# corr_Shawshank_Redemption = pd.DataFrame(similar_to_Shawshank_Redemption, columns=['Correlation'])
# corr_Shawshank_Redemption.dropna(inplace=True)
# print(corr_Shawshank_Redemption.head())

# corr_Silence_of_the_Lambs = pd.DataFrame(similar_to_Silence_of_the_Lambs, columns=['Correlation'])
# corr_Silence_of_the_Lambs.dropna(inplace=True)
# print(corr_Silence_of_the_Lambs.head())

corr_x=pd.DataFrame(similar_to_x, columns=['Correlation'])
corr_x.dropna(inplace=True)

# corr_Shawshank_Redemption = corr_Shawshank_Redemption.join(ratings['number_of_ratings'])
# corr_Silence_of_the_Lambs = corr_Silence_of_the_Lambs.join(ratings['number_of_ratings'])

corr_x=corr_x.join(ratings['number_of_ratings'])

# print(corr_Shawshank_Redemption.head())
# print(corr_Silence_of_the_Lambs.head())


# print(corr_Shawshank_Redemption[corr_Shawshank_Redemption['number_of_ratings'] > 10].sort_values(by='Correlation', ascending=False).head(10))
# print(corr_Silence_of_the_Lambs[corr_Silence_of_the_Lambs['number_of_ratings'] > 10].sort_values(by='Correlation', ascending=False).head(10))

# movies_Shawshank_Redemption=corr_Shawshank_Redemption[corr_Shawshank_Redemption['number_of_ratings'] > 10].sort_values(by='Correlation', ascending=False)
# movies_Silence_of_the_Lambs=corr_Silence_of_the_Lambs[corr_Silence_of_the_Lambs['number_of_ratings'] > 10].sort_values(by='Correlation', ascending=False)

movies_x=corr_x[corr_x['number_of_ratings']>10].sort_values(by='Correlation',ascending=False)
print("You can watch below movies")
print(movies_x.iloc[0:10,0])

# print("You can watch below movies upon choosing Shawshank Redemption")
# print(movies_Shawshank_Redemption.iloc[0:10,0])












