# Example use of logistic regression from the script logistic.py
from sklearn import datasets
from neededfunc import*
from logistic import logistic, logistic_fast, softmax

# download MNIST dataset
digits = datasets.load_digits()
# define inputs and labels
inputs = digits.images
labels = digits.target

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size, test_size=test_size)
print("Number of training images: ", len(X_train))
print("Number of test images: ", len(X_test))

Y_train_onehot = to_categorical_numpy(Y_train)
Y_test_onehot = to_categorical_numpy(Y_test)

epochs = 100
batch_size = 20
eta = 0.01
lmbd = 0.01

epo = [30, 50, 70, 100, 150, 200]
acc = []

for e in epo:
    beta_out = logistic(X_train, Y_train_onehot, e, batch_size, lmbd, eta)
    #prediction part
    logits = X_test.dot(beta_out)
    Y_proba = softmax(logits)
    y_predictlog = np.argmax(Y_proba, axis=1)
    print("Accuracy score on test set from logistic regression: ")
    print(accuracy_score_numpy(Y_test, y_predictlog))
    acc.append(accuracy_score_numpy(Y_test, y_predictlog))

plt.plot(epo, acc, label="Accuracy")
plt.title("Accuracies with log regression")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
