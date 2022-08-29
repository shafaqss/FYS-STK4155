#Here I am doing the ensemble methods on the data
from breast_cancer1 import*
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#Using a voting classifier
log_clf = LogisticRegression(solver="liblinear", random_state=0)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=0)
svm_clf = SVC(gamma="auto", C=0.2, random_state=0)
sgd_clf = SGDClassifier(random_state=0)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('sgd', sgd_clf)],
    voting='hard')

voting_clf.fit(X_train, y_train)

print("Ensemble methods")
print("-----------------")
print("Voting classifier")
print()
print("Hard classifier")
print("----------------")
for classf in (log_clf, rnd_clf, svm_clf, sgd_clf, voting_clf):
    classf.fit(X_train, y_train)
    y_pred = classf.predict(X_test)
    print(classf.__class__.__name__, accuracy_score(y_test, y_pred))
print()

log_clf = LogisticRegression(solver="liblinear", random_state=0)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=0)
svm_clf = SVC(gamma="auto",C=0.2, probability=True, random_state=0)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)

print("Soft classifier")
print("----------------")
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#using bagging
from sklearn.ensemble import BaggingClassifier
#max_samples=10

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=12), n_estimators=253,
    bootstrap=True, n_jobs=-1, random_state=12)

bag_clf.fit(X_train, y_train)
print()
print("Bagging")
print("Training set accuracy with bagging :", bag_clf.score(X_train, y_train))
print("Test set accuracy with bagging     :", bag_clf.score(X_test, y_test))

#using random forests
random_ff = RandomForestClassifier(n_estimators=253, random_state=0)
random_ff.fit(X_train, y_train)
print()
print("Random forests with 253 trees")
print("Training set accuracy     :", random_ff.score(X_train, y_train))
print("Test set accuracy         :", random_ff.score(X_test, y_test))
