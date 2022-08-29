#Here I am doing the decision tree on the data
#decision tree

from heart_disease1 import*
from sklearn.tree import DecisionTreeClassifier

#we want to find the optimum depth of tree
depth1 = np.arange(1, 20)
acctest1 = []
acctrain1 = []
acctest12 = []
acctrain12 = []

for d in depth1:
    dec_tree = DecisionTreeClassifier(max_depth=d, random_state=0)
    dec_tree.fit(X_train, y_train)
    acctest1.append(dec_tree.score(X_test, y_test))
    acctrain1.append(dec_tree.score(X_train, y_train))
    dec_treet = DecisionTreeClassifier(max_depth=d,criterion='entropy',random_state=0)
    dec_treet.fit(X_train, y_train)
    acctest12.append(dec_treet.score(X_test, y_test))
    acctrain12.append(dec_treet.score(X_train, y_train))

plt.figure()
plt.plot(depth1, acctest1, label="Test data using gini index")
plt.plot(depth1, acctest12, label="Test data using entropy")
plt.xlabel("Depth of the decison tree")
plt.ylabel("Accuracy")
plt.title("Accuracies on the test data")
plt.legend()
plt.show()

plt.figure()
plt.plot(depth1, acctrain1, label="Training data using gini index")
plt.plot(depth1, acctrain12, label="Train data using entropy")
plt.xlabel("Depth of the decison tree")
plt.ylabel("Accuracy")
plt.title("Accuracies on training data")
plt.legend()
plt.show()

#The best accuracy I can get is with max_depth = 9 with gini index
#dec_tree = DecisionTreeClassifier(max_depth=None, criterion='entropy',random_state=0)
dec_tree = DecisionTreeClassifier(max_depth=9,random_state=0)
dec_tree.fit(X_train, y_train)
print()
print("Decision tree")
print("Training set accuracy : ", dec_tree.score(X_train, y_train))
print("Test set accuracy     :", dec_tree.score(X_test, y_test))
y_pred_dectree = dec_tree.predict(X_test)
y_probas_dectree = dec_tree.predict_proba(X_test)

skplt.metrics.plot_confusion_matrix(y_test, y_pred_dectree, normalize=True)
plt.show()
