# Import Modules
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def load_dataset(to_print=0, x=0, y=0):
    dataset = datasets.load_iris()
    if to_print == 1:
        print_dataset_info(dataset)
    elif to_print == 2:
        xAxis = dataset.data[:, x]
        yAxis = dataset.data[:, y]
        plt.scatter(xAxis, yAxis, c=dataset.target)
        plt.show()

    return dataset


def print_dataset_info(dataset):
    print(dataset)
    print(dir(dataset))
    print(dataset.feature_names)
    print(dataset.target)
    print(dataset.target_names)
    print(dataset.filename)


def k_means_model(dataset):
    model = KMeans(n_clusters=3)    # we have 3 types of flowers, hence 3 clusters (aka groups)
    model.fit(dataset.data)  # only passing data without parameters
    predictions = model.predict(dataset.data)
    print("Accuracy is: ", accuracy_score(dataset.target, predictions))
    print("------------------------------------------------------------------")
    print(classification_report(dataset.target, predictions))


dataset = load_dataset()  # must choose columns from 0 to 3 if you're showing the scatter plot.
k_means_model(dataset)
