#Wisconsin breast cancer data analysis
#loading the data and splitting into train and test
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

bc_data = load_breast_cancer()
#Breast cancer data, X is the design matrix with shape (569, 30)
X = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
#print(X.head())
#Labels, y is the targets/labels with shape (569,)
y = pd.DataFrame(bc_data.target)
#print(y.head())
y = y.iloc[:,0]

#print(X.info())
print()
print("Data information")
print("Design matrix ", X.shape)
print("Targets", y.shape)

"""
#Correlation matrix
correlation_matrix = X.corr().round(1)
print("Shape of the correlation matrix: ", correlation_matrix.shape)
plt.figure(figsize=(14,7))
sns.heatmap(data=correlation_matrix, annot=True)
#plt.show()

#PCA analysis
pca = PCA().fit(X)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of features')
plt.ylabel('Explained variance')
plt.title("Explained variance as a function of the number of features")
#plt.show()
"""
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#now scale the data
scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print()
print("Data information ")
print("Training data")
print("X train ", X_train.shape)
print("y train ", y_train.shape)
print("Testing data")
print("X test", X_test.shape)
print("y test", y_test.shape)
print()
