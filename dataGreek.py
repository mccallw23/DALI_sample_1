import json
import csv
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


# 3D PCA:
data_frame = pd.read_csv("data_file_Greek_removed.csv")

# in this case, the last feature, "target", is encoded as identical to the 'affiliated' column, which is removed from the data.  After PCA is done, we will use affiliation as a target
# to color code our results and see if there are observable categorical differences between the other habits and characteristics of the affiliated and non affiliated people.
features = ['year', 'gender', 'heightInches', 'happiness', 'stressed', 'sleepPerNight', 'socialDinnerPerWeek','alcoholDrinksPerWeek', 'caffeineRating', 'numOfLanguages',
            'gymPerWeek', 'hoursOnScreen', 'phoneType']

# before doing feature separation, lets deal with missing data by imputation:
# In this case, I'm going to use the median value for every column.  The reason for this choice
# is that we are interested in categorizing distinct groups, so having null information should
# neither swing the data towards one group or another.  Missing data will thus have the result of pulling
# the person towards the "middle" of the data, diminishing a given person's "individuality".

data_frame['year'].fillna(data_frame['year'].median(), inplace = True)
data_frame['gender'].fillna(data_frame['gender'].median(), inplace = True)
data_frame['heightInches'].fillna(data_frame['heightInches'].median(), inplace = True)
data_frame['happiness'].fillna(data_frame['happiness'].median(), inplace = True)
data_frame['stressed'].fillna(data_frame['stressed'].median(), inplace = True)
data_frame['sleepPerNight'].fillna(data_frame['sleepPerNight'].median(), inplace = True)
data_frame['socialDinnerPerWeek'].fillna(data_frame['socialDinnerPerWeek'].median(), inplace = True)
data_frame['alcoholDrinksPerWeek'].fillna(data_frame['alcoholDrinksPerWeek'].median(), inplace = True)
data_frame['caffeineRating'].fillna(data_frame['caffeineRating'].median(), inplace = True)
#data_frame['affiliated'].fillna(data_frame['affiliated'].median(), inplace = True)
data_frame['numOfLanguages'].fillna(data_frame['numOfLanguages'].median(), inplace = True)
data_frame['gymPerWeek'].fillna(data_frame['gymPerWeek'].median(), inplace = True)
data_frame['hoursOnScreen'].fillna(data_frame['hoursOnScreen'].median(), inplace = True)
data_frame['phoneType'].fillna(data_frame['phoneType'].median(), inplace = True)


# Separating out the features
x = data_frame.loc[:, features].values
# Separating out the target data (which in this case is encoded as greek affiliation)
y = data_frame.loc[:,['target']].values
# Standardizing the features
# to do PCA we need to scale all features to have a mean of 0 and stdev of 1.
x = StandardScaler().fit_transform(x)

# we are going to identify 3 principal components, transform our data, and graph the results
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, data_frame[['target']]], axis = 1)

fig = plt.figure(figsize = (20,20))
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(['unnaffiliated','affiliated'])
ax.grid()
print("explained variance: ")
print(pca.explained_variance_ratio_)
plt.show()