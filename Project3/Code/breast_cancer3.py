#Doing logistic regression on the breast cancer data but with
#my own function that I used in project 2
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from neededfunc import*
from logistic import logistic, logistic_fast, softmax

bc_data = load_breast_cancer()
#Breast cancer data, X is the design matrix with shape (569, 30)
X = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
#print(X.head())
#Labels, y is the targets/labels with shape (569,)
y = bc_data.target #y has to be this variable for my own logistic function

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#now scale the data
scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#Trying my own logistic function
epochs = 500; batch_size = 20
eta = 0.2; lmbd = 0
beta_out = logistic(X_train, y_train, epochs, batch_size, lmbd, eta)
#prediction part
logits = X_test.dot(beta_out)
Y_proba = softmax(logits)
y_predictlog = np.argmax(Y_proba, axis=1)
print()
print("Logistic regression with my own code")
print("Accuracy score on test set from logistic regression: ")
print(accuracy_score(y_test, y_predictlog))
