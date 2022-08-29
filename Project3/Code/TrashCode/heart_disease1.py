#South Africa Coronary heart disease analysis
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

heart = pd.read_csv('SAHeart.txt', header=0)
heart['famhist'].replace("Present", 1, inplace=True)
heart['famhist'].replace("Absent", 0, inplace=True)

heart = heart.drop("row.names", axis=1)
#heart.info()
#print(heart.head())
print()

#Design matrix with size (462, 9)
X = heart.iloc[:,:9]
#targets chd or no chd, with size (462,)
y = heart.iloc[:,9] #targets

print("Data information")
print("Design matrix ", X.shape)
print("Targets", y.shape)

"""
#Correlation matrix
correlation_matrix = X.corr().round(1)
print("Shape of the correlation matrix: ",correlation_matrix.shape)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

#PCA analysis
pca = PCA().fit(X)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of features')
plt.ylabel('Explained variance')
plt.title("Explained variance as a function of the number of features")
plt.show()
"""
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

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
