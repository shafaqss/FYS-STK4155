# Example use of NN_Classification class
from sklearn import datasets
from neededfunc import*
from NeuralNetwork_Class import NN_Classification
import activations as act

# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target
print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))
# flatten the image
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size, test_size=test_size)
print("Number of training images: ", len(X_train))
print("Number of test images: ", len(X_test))
print()
print("training info")
print("train design matrix ",X_train.shape)
print("train z ",Y_train.shape)
print()
print("test info")
print("test design matrix ",X_test.shape)
print("test z ",Y_test.shape)

Y_train_onehot = to_categorical_numpy(Y_train)
Y_test_onehot = to_categorical_numpy(Y_test)

nepoch = 100
batch_size = 30
eta = 0.01
lmbd = 0.01
n_hidden_neurons = 50
n_categories = 10 #calssifying images from 0-9

nn2 = NN_Classification(X_train, Y_train_onehot, act.Sigmoid(), eta=eta, lmbd=lmbd,
                    epochs=nepoch, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons,
                    n_outputs=n_categories)

nn2.train()
test_predict = nn2.predict(X_test)
print()
print("Accuracy score on test set from NN on MNIST data: ")
print(accuracy_score_numpy(Y_test, test_predict))
