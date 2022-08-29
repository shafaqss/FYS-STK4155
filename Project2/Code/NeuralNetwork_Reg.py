import activations as act
import costs as cost_function
from neededfunc import*

class IndividualLayer:
    def __init__(self, n_inputs, n_outputs, activation):
        """Initializer for IndividualLayer, n_input and n_outputs is the
        size of the layer, activation is the type of activation function
        you wish to use in the particular layer."""
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.weights = np.random.randn(self.n_inputs, self.n_outputs)
        self.bias = np.random.randn(1, self.n_outputs)

    def __call__(self, X):
        """Performs the calculaions in each layer, that is, it performs the
        operation activation(wx + b) = activation(z). It also calculates the
        derivative activation'(z)"""
        self.z = X @ self.weights + self.bias
        self.a = self.activation(self.z)
        self.a_derivative = self.activation.prime(self.z) #needed for backpropagation
        return self.a

class NN_Regression:
    def __init__(self, n_inputs, neurons, n_outputs, cost_function):
        """Creates the Neural Network for regression, n_inputs and n_outputs is as
        before, neurons is the number of neurons in a layer and cost function is the
        type of cost function you wish to minimize, in regression it is set to
        the mean squarred error(MSE)"""
        self.n_inputs = n_inputs
        self.neurons = neurons
        self.n_outputs = n_outputs
        self.cost_function = cost_function

    def create_network(self, activation, output_activation):
        """Activation is the type of activation function is the hidden layers, while
        output_activation is the activation function that will be used in the
        output layer only"""
        self.layers = [] #store the layers in a list

        # first we add the input layer
        self.layers.append(IndividualLayer(self.n_inputs, self.neurons[0], activation)) # input layer
        # add then the hidden layers
        for i in range(len(self.neurons) - 1):
            self.layers.append(IndividualLayer(self.neurons[i], self.neurons[i + 1], activation)) # hidden layers

        # finally adding the output layer
        self.layers.append(IndividualLayer(self.neurons[-1], self.n_outputs, output_activation)) # output layer

    def feed_forward(self, a):
        """Perform the feed forward to reach the output layer, layer L."""
        for layer in self.layers:
            a = layer(a)

    def back_propagation(self, X, y, eta, lmbda):
        """Perform the back propogation to find the errors in each layer and update
        the weights and bias accordingly. Here eta is the learning rate, and
        lmbda is the regularization parameter."""
        self.feed_forward(X)
        #Start calculating the errors in the output layer
        L = self.layers
        cost_der = self.cost_function.prime(L[-1].a, y) #partial derivative of cost w.r.t a
        delta_L = cost_der * L[-1].a_derivative #error in the last layer

        #finding the error in each layer, starting from layer L-1
        for l in reversed(range(1, len(L) - 1)):
            delta_L = (delta_L @ L[l + 1].weights.T) * L[l].a_derivative
            #update the weights
            L[l].weights =  L[l].weights - eta*(L[l - 1].a.T @ delta_L) - 2*eta*lmbda*L[l].weights
            #update the biases
            L[l].bias = L[l].bias - eta*delta_L[0, :]

        #finally updating the weights and biases in the first hidden layer
        delta_L = (delta_L @ L[1].weights.T) * L[0].a_derivative
        L[0].weights = L[0].weights - eta*(X.T @ delta_L) - 2*eta*lmbda*L[0].weights
        L[0].bias = L[0].bias - eta*delta_L[0, :]

if __name__ == '__main__':
    # Example usage of this class
    #Regression on franke function
    n = 100 #number of samples
    degree = 5 #degree of polynomial
    noise = 0.01
    noise_array = np.random.normal(0,0.1,n)*noise

    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    x, y = np.meshgrid(x,y)
    X = create_design_matrix(x, y, degree)
    print("the shape of X is", X.shape)
    z = (FrankeFunction(x, y) + noise_array).ravel()
    z = z.reshape(-1,1)
    print("the shape is z is", z.shape)
    X_train, X_test, z_train, z_test = train_test_np(X, z)

    ninputs = X_test.shape[1]
    neurons = [20, 50, 30]
    noutputs = 1 #1 category in regression case

    nepoch = 100
    eta = 0.01 #learning rate
    lmbda = 0.316 #regularization parameter
    mse_nn = np.zeros(nepoch)

    nn1 = NN_Regression(ninputs, neurons, noutputs, cost_function.MSE())
    nn1.create_network(act.Sigmoid(), act.Identity())
    nn1.feed_forward(X_test)
    print(f"MSE before training has occured: {MSE(z_test, nn1.layers[-1].a)}")

    for i in range(nepoch):
        nn1.back_propagation(X_train, z_train, eta, lmbda)
        nn1.feed_forward(X_test)
        mse_nn[i] = MSE(z_test, nn1.layers[-1].a)

    plt.plot(np.arange(nepoch), mse_nn, label="lambda = 0.316")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Ridge regression on franke with neural network")
    plt.legend()
    plt.show()

    nn1.feed_forward(X_test)
    print(f"MSE on test set after training: {MSE(z_test, nn1.layers[-1].a)}")
