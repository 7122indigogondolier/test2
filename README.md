# Test 2: Predicting Chance of Admission 
### Team: Utkrist P. Thapa '21, Abhi Jha '21, Tina Jin '21

### Exploratory Data Analysis 
We first identified the features of our model which were GRE score, TOEFL score, university rating, statement of purpose rating, letter of recommendation rating, cumulative GPA, research, socioeconomic status percentage, and race. The chance of admission is the output we are trying to predict. Note that serial number is omitted from the features because it is just for our reference and should not matter in predicting chance of admission. Then we noticed that the feature 'RACE' do not contain numerical values. We used onehot encoding to transform this feature column into four columns denoting asian, african american, latinx and white respectively.

We ran isna() on the entire dataframe, and found that there are 44 rows containing missing values, and there are 356 rows of valid data in total. Instead of emulating the data, we chose to drop rows that contain at least one missing value, because the rows dropped made only a small portion of the total data points.

Since the measurements between each feature are different, we need to scale the features before applying them to a model. We used the standard scaler (WHY?)

Then we ran a statistical describe() on the target variable to see the distribution of the output. Note that this target variable is continuous, so we tried both regression and binary classification using a threshold (chance of admission and whether someone would be admitted). Looking at the histogram, we would say that the target variable is not skewed. If we were to run a binary classifier, the two classes would be balanced using the threshold 0.73, because it was close to the mean and median of the dataset.

Furthermore, we ran a scatter plot matrix and heat map to visualize the given dataset. 
![](/images/xxxfigure)

### Model Selection
For regression, we chose to use linear regression and random forest regressor.
For binary classification, we tried logistic regression with stochastic gradient descent, KNN, decision tree/random forest, SVM and Naive Bayes.

### Evaluating the Models
![](/images/somemetrics)


