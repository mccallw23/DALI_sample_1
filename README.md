# Will McCall
DALI Challenge 1:  Data Challenge

The different genders represented at the DALI lab do not have a discernible difference in overall tendencies accross 14 survey datapoints, but people of greek affilation within the lab actually do exhibit a more discernible set of characteristics in comparrison to thier non greek counterparts.

For this challenge, I decided to take the dataset containing anonymized data of 65 DALI members, each of whom reported 14 pieces of data about themselves:

"year"
"gender"
"heightInches"
"happiness": 
"stressed"
"sleepPerNight"
"socialDinnerPerWeek" 
"alcoholDrinksPerWeek"
"caffeineRating"
"affiliated":
"numOfLanguages"
"gymPerWeek":
"hoursOnScreen"
"phoneType"

The dataset is available at: https://github.com/dali-lab/dali-challenges/blob/master/data/DALI_Data-Anon.json

Using principal component analysis, I reduced the dimensionality of this data from 14 to 3 allowing the 14D data to be graphed in 3D with minimum loss of information.  I also color coded the data first by gender and then by greek affiliation to determine if either set of characteristics exhibitied clustering.  With respect to gender there was a lot of overlap in the data, suggesting that the different genders represented at the DALI lab do not have categorically different self reported habits and characteristics, but affiliated vs non affiliated members actually do!

Said differently, it should be difficult to infer a person's gender from the other data they report, but should be more possible to infer thier greek affiliaton given the same information.

# imputation

prior to performing some computations, such as PCA reduction, it was necessary to fill in missing data, which was done through imputation by the median.  This method was chosen for its characteristic of dampening the outlier effect of whatever data it was applied to, which I decided would be best on the basis that we should assume a person exhibits typical characteristics of the dataset as a whole unless we are told otherwise.  This compromise is not perfect but permits a sufficiently reasonable degree of accuracy given its approximate correctness and the relatively few instances of missing data it was used for.

# Other considerations

In order to perform greek life analysis, I had to also exclude 22's from the data, who at the time of recording, were dissalowed from affiliation and thus not applicable.

In order to perform both analyses, I had to remove the target attribute from dimensionality reduction so that it did not cause unwarrented clustering of the data along principal component axes.


# dataGender.py

.. includes the code used to graph the PCA data with respect to gender.  This could trivially be changed to graph with respect to affiliation, or any other one of the 14 categories.

# dataGreek.py:

... includes the code to graph PCA data with respect to affiliation

# data_file_gender_removed.csv:

... contains all of the data except for gender (which should be excluded from analysis so it doesn't artifically cluster the points).  It includes a target column identical to gender which is not considered in dimensionality reduction, but is later used for color coding.

# data_file_greek_removed.csv:

... contains all of the data except for affiliation (which should be excluded from analysis so it doesn't artifically cluster the points).  It includes a target column identical to greek affiliation which is not considered in dimensionality reduction, but is later used for color coding.

# Folders

The included folders showcase the 3D data rendered from different angles.
