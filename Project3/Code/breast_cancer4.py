#Here I am doing the decision tree on the data
#decision tree
from breast_cancer1 import*
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier(max_depth=3, random_state=0)
dec_tree.fit(X_train, y_train)
print("Decisin tree")
print("Training set accuracy ", dec_tree.score(X_train, y_train))
print("Test set accuracy ", dec_tree.score(X_test, y_test))

#want to find the minimum sample leaves
leaf = np.arange(1, 20)
acctest2 = []
acctrain2 = []
acctest22 = []
acctrain22 = []

for l in leaf:
    dec_tree = DecisionTreeClassifier(min_samples_leaf=l,random_state=0)
    dec_tree.fit(X_train, y_train)
    acctest2.append(dec_tree.score(X_test, y_test))
    acctrain2.append(dec_tree.score(X_train, y_train))
    dec_treett = DecisionTreeClassifier(min_samples_leaf=l,criterion='entropy', random_state=0)
    dec_treett.fit(X_train, y_train)
    acctest22.append(dec_treett.score(X_test, y_test))
    acctrain22.append(dec_treett.score(X_train, y_train))

plt.figure()
plt.plot(leaf, acctest2, label="Test data using gini index")
plt.plot(leaf, acctest22, label="Test data using entropy")
plt.xlabel("Minimum number of samples at a leaf node")
plt.ylabel("Accuracy")
plt.title("Accuracies on test data")
plt.legend()

plt.figure()
plt.plot(leaf, acctrain2, label="Train data using gini index")
plt.plot(leaf, acctrain22, label="Train data using entropy")
plt.xlabel("Minimum number of samples at a leaf node")
plt.ylabel("Accuracy")
plt.title("Accuracies on train data")
plt.legend()
plt.show()
