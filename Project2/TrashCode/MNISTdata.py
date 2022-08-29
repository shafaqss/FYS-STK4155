from sklearn import datasets
from neededfunc import*
#from sklearn.metrics import accuracy_score
from NeuralNetwork_Class import NN_Classification
import activations as act
from logistic import logistic, logistic_fast, softmax

np.random.seed(0)
# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))

# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))
print("labels shape, ", labels.shape)

"""
# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)
for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
plt.show()
"""
# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size, test_size=test_size)
print("Number of training images: ", len(X_train))
print("Number of test images: ", len(X_test))

Y_train_onehot = to_categorical_numpy(Y_train)
Y_test_onehot = to_categorical_numpy(Y_test)
#print("onehot shape is ",Y_train_onehot.shape)

#epochs = 100
batch_size = 20
eta = 0.01
lmbd = 0.2
n_hidden_neurons = 40
n_categories = 10 #calssifying images from 0-9
n_iterations = 200

ep = [40, 80, 100, 120, 160, 200, 220]
accnn = []
for e in ep:
    nn1 = NN_Classification(X_train, Y_train_onehot, act.Sigmoid(), eta=eta, lmbd=lmbd,
                        epochs=e, batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons,
                        n_outputs=n_categories)
    nn1.train()
    test_predictnn = nn1.predict(X_test)
    accnn.append(accuracy_score_numpy(Y_test, test_predictnn))
    #print("Accuracy score on test set from neural network: ")
    #print(accuracy_score_numpy(Y_test, test_predictnn))
    #print()

plt.plot(ep, accnn)
plt.title("Accuracies with MNIST using Sigmoid activation")
plt.xlabel("Epochs")
plt.ylabel("Accuracies")
plt.show()

#epochs = 100
batch_size = 20
eta = 0.01
lmbd = 0.2
n_hidden_neurons = 40
n_categories = 10 #calssifying images from 0-9
#n_iterations = 200

iter = [50, 80, 100, 150, 200]
acclog = []
for i in iter:
    out = logistic_fast(X_train, Y_train_onehot, i, lmbd, eta)
    #prediction part
    logits = X_test.dot(out)
    Y_proba = softmax(logits)
    y_predictlog = np.argmax(Y_proba, axis=1)
    acclog.append((accuracy_score_numpy(Y_test, y_predictlog)))

plt.plot(iter, acclog)
plt.title("Accuracies with MNIST using logistic regression")
plt.xlabel("Number of iterations")
plt.ylabel("Accuracies")
plt.show()

"""
#logistic(X_train, y_train, n_epochs, bsize, lmbd, eta
#logistic_fast(X_train, y_train, n_iterations, lmbd, eta)
#out = logistic_fast(X_train, Y_train_onehot, n_iterations, lmbd, eta)
out = logistic(X_train, Y_train_onehot, epochs, batch_size, lmbd, eta)
#prediction part
logits = X_test.dot(out)
Y_proba = softmax(logits)
y_predictlog = np.argmax(Y_proba, axis=1)
print("Accuracy score on test set from logistic regression: ")
print(accuracy_score_numpy(Y_test, y_predictlog))
"""
