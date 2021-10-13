# Will McCall
DALI Challenge 1:  Data Challenge

For this challenge, I decided to take the dataset containing anonymized data of 66 DALI members, each of whom reported 14 pieces of data about themselves:

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

Using principal component analysis, I reduced the dimensionality of this data from 14 to 3 allowing the 14D data to be graphed in 3D with minimum loss of information.  I also color coded the data first by gender and then by greek affiliation to determine if either set of characteristics exhibitied clustering.  They both did.  What this seems to imply is that the different genders represented at the DALI lab have categorically different self reported habits and characteristics, and the same can be said for affiliated vs non affiliated members.

Said differently, based on the data, it should be possible to infer a person's gender or greek affiliaton from the data with a reasonable degree of accuracy.

# dataviz.py

.. includes the code used to graph the dimensionally reduced data with respect to gender.  This could trivially be changed to graph with respect to affiliation, or any other one of the 14 categories.

# imputation

prior to performing some computations, such as PCA reduction, it was necessary to fill in missing data, which was done through imputation by the median.  This method was chosen for its characteristic of dampening the outlier effect of whatever data it was applied to, which I decided would be best on the basis that we should assume a person exhibits typical characteristics of the dataset as a whole unless we are told otherwise.  This compromise is not perfect but permits a sufficiently reasonable degree of accuracy given its approximate correctness and the relatively few instances of missing data it was used for.



