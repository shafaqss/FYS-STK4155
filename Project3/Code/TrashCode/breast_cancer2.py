from breast_cancer1 import*
from sklearn.linear_model import LogisticRegression

#Logistic regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
s1 = logreg.score(X_train, y_train)
s2 = logreg.score(X_test, y_test)

print("Logistic regression")
print("Training set accuracy with sklearn :", s1 )
print("Test set accuracy with sklearn     :", s2 )

#using only the principle components
pca = PCA(n_components = 5)
PC_train = pca.fit_transform(X_train)
logpca = LogisticRegression(solver='lbfgs')
logpca.fit(PC_train, y_train)
print("Training set accuracy with PCA data :", logpca.score(PC_train, y_train))

y_pred_log_reg = logreg.predict(X_test)
y_probas_log_reg = logreg.predict_proba(X_test)

plt.figure()
skplt.metrics.plot_confusion_matrix(y_test, y_pred_log_reg, normalize=True, cmap='YlOrBr')

plt.figure()
skplt.metrics.plot_roc(y_test, y_probas_log_reg)

plt.figure()
skplt.metrics.plot_cumulative_gain(y_test, y_probas_log_reg)

plt.figure()
skplt.metrics.plot_precision_recall(y_test, y_probas_log_reg)
#plt.show()
