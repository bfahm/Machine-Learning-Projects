# Import Modules
import sys
import scipy
import numpy
import matplotlib
import sklearn
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


seed = 7  # Retrieve the same results every run
scoring = 'accuracy'  # Test model score by accuracy (in %percentage)


# Check the versions of libraries
def check_versions():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))


def load_dataset(to_print=-1):
    """
    Loads a csv file into a dataFrame
    column = 'names'
    rows from 1 to 150.
    Takes a value from 1 to 8 to peek at the data in a special form or by printing out a graph,
    If left plank, it will just load the set.
    """

    url = "iris_data.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)

    if to_print == 0:  # Print the set as is.
        print(dataset.values)
    elif to_print == 1:  # Default print way
        print(dataset)
    elif to_print == 2:  # Peek at the first 20 elements
        print(dataset.head(20))
    elif to_print == 3:  # Print some info of the dataset
        print(dataset.describe())
    elif to_print == 4:
        print(dataset.groupby('class').count())
    elif to_print == 5:  # Box and whiskers plot
        '''
        The plot show a box, where the first edge represents the median of teh first half of the dataset,
        the second edge represents the median of the second half of the dataset,
        the middle edge (the one cutting the box into two pieces) represents the median of the dataset.
        The top and bottom lines (the whiskers) represents the rest of the dataset. 
        Refer to: https://www.youtube.com/watch?v=09Cx7xuIXig for more.
        '''
        dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
        plt.show()
    elif to_print == 6:
        dataset.hist()
        plt.show()
    elif to_print == 7:  # scatter plot matrix
        '''
        Scatter plot matrix plots the relation between two pairs of attributes.
        Great for spotting a correlation between attribs.
        '''
        scatter_matrix(dataset)
        plt.show()
    elif to_print == 8:  # just print the shape.
        print("Shape = ", dataset.shape)

    return dataset.values


def split_data(test_size=0.2):
    """
    Splits the dataset into portions, training set and test set.
    Test set is used to provide "unseen" examples to the model, aka, examples where the model didn't train on to test
    its performance.
    :param test_size: takes the ratio to which data should be slitted, defaulted at 0.2 (20%)
    :return: X_train, Y_train, X_test, Y_test
    """

    data = load_dataset()
    X = data[:, 0:4]  # Take columns 0, 1, 2, 3 (excluding the 'class' column which should be output) as inputs
    Y = data[:, 4]  # Take the last column, 'class', as the output

    # random_state shuffles the set, keeping it to "seed" makes sure it results the same every run.
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

    return X_train, X_test, Y_train, Y_test


def compare_models(X_train, Y_train):
    # Define a list of models where we compare between.
    models = [('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')),
              ('Decision Tree', DecisionTreeClassifier()),
              ('Gaussian Naive Bayes', GaussianNB()),
              ('Support Vector Machines', SVC(gamma='auto'))]

    for name, model in models:
        kFold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_score = model_selection.cross_val_score(model, X_train, Y_train, scoring=scoring, cv=kFold)
        print(name, ": ", cv_score.mean())


def logistic_model(X_train, X_test, Y_train, Y_test):
    classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)

    print("Accuracy is: ", accuracy_score(Y_test, predictions))
    print("------------------------------------------------------------------")
    print(classification_report(Y_test, predictions))


# check_versions()
X_train, X_test, Y_train, Y_test = split_data()
compare_models(X_train, Y_train)
logistic_model(X_train, X_test, Y_train, Y_test)
