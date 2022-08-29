from heart_disease1 import*
from sklearn.linear_model import LogisticRegression
# Logistic Regression

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
s1 = logreg.score(X_train, y_train)
s2 = logreg.score(X_test, y_test)

print("Logistic reggession")
print("Training set accuracy :", s1 )
print("Test set accuracy     :", s2 )

#using only the principle components
pca = PCA(n_components = 4)
PC_train = pca.fit_transform(X_train)
logpca = LogisticRegression(solver='lbfgs')
logpca.fit(PC_train, y_train)
print("Training set accuracy with PCA data:", logpca.score(PC_train, y_train))

y_pred_log_reg = logreg.predict(X_test)
y_probas_log_reg = logreg.predict_proba(X_test)

skplt.metrics.plot_confusion_matrix(y_test, y_pred_log_reg, normalize=True, cmap='YlOrBr')
#plt.show()
skplt.metrics.plot_roc(y_test, y_probas_log_reg)
#plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas_log_reg)
#plt.show()
skplt.metrics.plot_precision_recall(y_test, y_probas_log_reg)
#plt.show()
