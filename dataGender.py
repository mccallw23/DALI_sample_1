import json
import csv

import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

# during data cleaning and reformatting, the following changes were made from the original JSON file:
# 1) the apostrophe was removed from the year column.
# 2) The gender column was moved to the end of the frame and renamed as the 'target' so as not to interfere with principal component analysis
# 3) in this column, male is represented as  0, female as 1, other as -1
# 4) ios is 0, andriod 1, other -1
# 5) when missing data was found, I chose to impute by the median (more information later)

# additional notes:  3 dimensional PCA accounted for 40% of the variance in the data.  This means PCA created 3 new features
# which captured as much data as approximately 6/14 of the original features.  However, it also means that a significant amount
# of variance is left unaccounted for.

# 3D PCA:
data_frame = pd.read_csv("data_file_gender_removed.csv")


# in this case, the last feature, "target", is encoded as identical the gender column.  After PCA is done, we will use gender as a target
# to colorcode our results and see if there are observable categorical differences between the other habits and characteristics of the genders
# this also ensures that gender does not play a direct role in distinguishing principal components in the dataset
features = ['year','heightInches', 'happiness', 'stressed', 'sleepPerNight', 'socialDinnerPerWeek','alcoholDrinksPerWeek', 'caffeineRating', 'affiliated', 'numOfLanguages',
            'gymPerWeek', 'hoursOnScreen', 'phoneType']

# before doing feature separation, lets deal with missing data by imputation:
# In this case, I'm going to use the median value for every column.  The reason for this choice
# is that we are interested in categorizing distinct groups, so having null information should
# neither swing the data towards one group or another.  Missing data will thus have the result of pulling
# the person towards the "middle" of the data, diminishing a given person's "individuality".

data_frame['year'].fillna(data_frame['year'].median(), inplace = True)
data_frame['heightInches'].fillna(data_frame['heightInches'].median(), inplace = True)
data_frame['happiness'].fillna(data_frame['happiness'].median(), inplace = True)
data_frame['stressed'].fillna(data_frame['stressed'].median(), inplace = True)
data_frame['sleepPerNight'].fillna(data_frame['sleepPerNight'].median(), inplace = True)
data_frame['socialDinnerPerWeek'].fillna(data_frame['socialDinnerPerWeek'].median(), inplace = True)
data_frame['alcoholDrinksPerWeek'].fillna(data_frame['alcoholDrinksPerWeek'].median(), inplace = True)
data_frame['caffeineRating'].fillna(data_frame['caffeineRating'].median(), inplace = True)
data_frame['affiliated'].fillna(data_frame['affiliated'].median(), inplace = True)
data_frame['numOfLanguages'].fillna(data_frame['numOfLanguages'].median(), inplace = True)
data_frame['gymPerWeek'].fillna(data_frame['gymPerWeek'].median(), inplace = True)
data_frame['hoursOnScreen'].fillna(data_frame['hoursOnScreen'].median(), inplace = True)
data_frame['phoneType'].fillna(data_frame['phoneType'].median(), inplace = True)


# Separating out the features
x = data_frame.loc[:, features].values
# Separating out the target (which in this case is encoded as gender)
y = data_frame.loc[:,['target']].values
# Standardizing the features
# to do PCA we need to scale all features to have a mean of 0 and stdev of 1.
x = StandardScaler().fit_transform(x)

# we are going to identify 3 principal components, transform our data, and graph the results
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
# after the data has been transformed its ok to concatenate the gender data as a target
finalDf = pd.concat([principalDf, data_frame[['target']]], axis = 1)

# code for graphing the results:
fig = plt.figure(figsize = (20,20))
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = [-1, 0, 1]
colors = ['r', 'g', 'b']
# encode the target points with their correct colors, graph with respect to PCA
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target']== target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(['other','male','female'])
ax.grid()
# how much variance is explained by these principal components?
# print(indicesToKeep)
print("explained variance: ")
print(pca.explained_variance_ratio_)
plt.show()