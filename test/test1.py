import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import sys
import os

import time

start_time = time.time()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tree_influence.explainers import BoostIn

# load iris data
data = load_iris()
X, y = data['data'], data['target']

# use two classes, then split into train and test
idxs = np.where(y != 2)[0]
X, y = X[idxs], y[idxs]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# train GBDT model
model = LGBMClassifier().fit(X_train, y_train)

# fit influence estimator
explainer = BoostIn().fit(model, X_train, y_train)

# estimate training influences on each test instance
influence = explainer.get_local_influence(X_test, y_test)  # shape=(no. train, no. test)

# extract influence values for the first test instance
values = influence[:, 0]  # shape=(no. train,)

# sort training examples from:
# - most positively influential (decreases loss of the test instance the most), to
# - most negatively influential (increases loss of the test instance the most)
training_idxs = np.argsort(values)[::-1]
print(values[0])
index = training_idxs[0]
print(f"Most influential data index: {index}")
print(X_train[index])
print(y_train[index])
print(f"The entire array: {training_idxs}")
print(f"The entire values array: {values}")
end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")