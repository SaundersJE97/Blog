# Import the Iris dataset from sklearn
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets as ds
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt



iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(y_train)
y_train_index = np.argmax(pd.get_dummies(y_train).to_numpy(), axis=1)
y_test_index = np.argmax(pd.get_dummies(y_test).to_numpy(), axis=1)

# PCA visulisation
pca = PCA(n_components=2)
x_scaled = preprocessing.scale(X_train.to_numpy()[:,:-1])
x_reduced = pca.fit_transform(x_scaled)

plt.scatter(x_reduced[:,0][y_train_index == 0], x_reduced[:, 1][y_train_index == 0], c="blue", label="setosa", marker="+")
plt.scatter(x_reduced[:,0][y_train_index == 1], x_reduced[:, 1][y_train_index == 1], c="orange", label="versicolor", marker="*")
plt.scatter(x_reduced[:,0][y_train_index == 2], x_reduced[:, 1][y_train_index == 2], c="red", label="virginica", marker=".")
plt.legend()
#plt.show()

# Calculate entropy
def calculate_entropy(y):
    """Finds the unique elements of an array, return_counts returns the number of times each unique item appears"""
    a = np.unique(y, return_counts=True)

    #Probability of each unique item using the total entries of the array
    probability = np.divide(a[1].astype(float), np.sum(a[1].astype(float)))

    # Entropy of the data
    entropy = -np.sum(probability * np.log2(probability))
    return entropy

# Finding the split
def find_split(x, y):
    """Given a datastet and it's target values, this finds the optimal combination of feature and split point
    that gives the maximum information gain"""

    # starting entropy so we can measurement improvement...
    start_entropy = calculate_entropy(y)

    # Best thus far, initialised to a dud that will be replaced immediately
    best = {'infogain' : -np.inf}

    # Loop through all classes in the dataset
    for i in range(x.shape[1]):
        # Loop through each data point for each class, taking the data point as the split value.
        for split in np.unique(x[:, i]):
            # split the data via the split value for each data point in class 'i'
            left_side = x[:,i] <= split
            right_side = x[:,i] > split

            left_ind = np.sum(left_side)
            right_ind = np.sum(right_side)

            entropyLeft = (left_ind / len(y)) * calculate_entropy(y[left_side])
            entropyRight = (right_ind / len(y)) * calculate_entropy(y[right_side])

            infogain = start_entropy - entropyLeft - entropyRight

            if infogain > best['infogain']:
                best = {'feature' : i,
                        'split' : split,
                        'infogain' : infogain,
                        'left_indices' : np.where(left_side)[0],
                        'right_indices' : np.where(right_side)[0]}
    return best

def build_tree(x, y, max_depth = np.inf):
    # Check if either of the stopping conditions and if so generate a leaf node
    if max_depth == 1 or (y==y[0]).all():
        # generate leaf node
        classes, counts = np.unique(y, return_counts=True)
        return {'leaf' : True, 'class' : classes[np.argmax(counts)]}
    else:
        move = find_split(x, y)

        left = build_tree(x[move['left_indices'], :], y[move['left_indices']], max_depth-1)
        right = build_tree(x[move['right_indices'], :], y[move['right_indices']], max_depth-1)

        return {'leaf' : False,
                'feature' : move['feature'],
                'split' : move['split'],
                'infogain' : move['infogain'],
                'left' : left,
                'right' : right}


def predict(tree, samples):
    """Predicts class for every entry of a data matrix"""
    ret = np.empty(samples.shape[0], dtype=int)
    ret.fill(-1)
    indices = np.arange(samples.shape[0])

    def traverse(node, indices):
        nonlocal samples
        nonlocal ret

        if node['leaf']:
            ret[indices] = node['class']

        else:
            going_left = samples[indices, node['feature']] <= node['split']
            left_indices = indices[going_left]
            right_indices = indices[np.logical_not(going_left)]

            if left_indices.shape[0] > 0:
                traverse(node['left'], left_indices)

            if right_indices.shape[0] > 0:
                traverse(node['right'], right_indices)

    traverse(tree, indices)
    return ret


tree = build_tree(X_train.to_numpy(), y_train_index, 4)

prediction_train = predict(tree, X_train.to_numpy())
train_accuracy = np.sum(prediction_train == y_train_index) / len(y_train_index)

prediction_test = predict(tree, X_test.to_numpy())
test_accuracy = np.sum(prediction_test == y_test_index) / len(y_test_index)

print("Accuracy: ", train_accuracy, test_accuracy)