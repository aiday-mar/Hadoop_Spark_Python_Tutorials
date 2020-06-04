# Applied Machine Learning : Algorithms

The full dataset needs to be split into the training data, the validation data and the testing data. The training data then needs to be split into the fivefold cross-validation data. We fit an initial model and evaluate, then we tune the hyperparameters, evaluate on the validation set and select and evelutate the final model on the test set. We are going to read in data using python :

```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline

titanic = pd.read_csv('../../../titanic.csv')
titanic.head()
titantic.isNull().sum()  // sums the number of times we have absent data for the feature

// suppose you want to fill in the missing values in the column with the mean value of that column then 
titanic['Age'].fillna(titanic['Age'].mean(), inplace = True)
titanic.head(10)

// now suppose that we  want to print out a categorical plot, then we can write 
// the size of the bar corresponds to the sample size
for i, col in enumerate(['SibSp','Parch']):
  plt.figure(i)
  sns.catplot(x=col, y='survived',  data=titanic, kind='point', aspect=2)

// we can combine the two plots together
titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']
titanic.drop(['PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)
titanic.groupby(titanic)

// we can then perform some calculations by grouping data together
// here we group by according to where the column 'cabin' has null values, and where it hasn't. Then we take the mean of the 
// corresponding entries in the 'Survived' column
titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean()

// where we have different properties 
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
titanic.head()

// if you want to change some values to numerical values, like male to zero and female to one then you can write 
gender_num = {'male' : 0, 'female' : 1}
titanic['Sex'] = titanic['Sex'].map(gender_num)
titanic.head()

// we are going to split the data into training and testing data
features = titanic.drop('Survived', axis=1)
labels = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)
```

Regression is a statistical process for estimating the relationships among variables, often to make a prediction about some outcome. Logistic regression is a form of regression where the target variable is binary. The C hyperparameter is a regularization parameter in logistic regression that controls how closely the model fits to the training data. Regularization is a technique used to reduce overfitting by discouraging overly complex models in some way. We have the following code :

```
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('../../../train_features.csv')
tr_labels = pd.read_csv('../../../train_labels.csv', header=None)

lr = LogisticRegression()
parameters = {
  'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

// the below outputs C = 1
cv.best_estimator_
```

A support vector machine is a classifier that finds an optimal hyperplane that maximizes the margin between two classes. The kernel trick transforms data that is not linearly separable in an n-dimensional space to a higher dimension where it is linearly separable.
