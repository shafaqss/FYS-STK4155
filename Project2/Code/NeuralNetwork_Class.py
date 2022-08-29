import activations as act
from neededfunc import*

class NN_Classification:
    def __init__(self, X_data, Y_data, activation, n_hidden_neurons,
            n_outputs, epochs, batch_size, eta, lmbd):
        """Creates the Neural Network for classification, this NN has one input
        layer, one hidden layer, and one output layer.

        X_data: The input X data to be trained on
        Y_data: The input y data
        activation: The type of activation function used inthe hidden layer
        n_hidden_neurons: number of nodes in the hidden layer
        n_outputs: number of categories in the output layer
        epochs : number of epochs to use when training
        batch_size: size of batches when training
        eta : learning rate
        lmbd: regularization parameter
        """
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.activation = activation
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        #start by initializing the weights and biases
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """Initializing the weights and biases to start the computations. Sets
        the weights and biases to the correct sizes"""
        #weights and biases for the hidden layer
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.random.randn(self.n_hidden_neurons) + 0.01

        #weights ad biases for the output layer
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_outputs)
        self.output_bias = np.random.randn(self.n_outputs) + 0.01

    def feed_forward(self):
        """Perform the feed forward when training to reach the output layer."""
        #computing activation(wx + b) for the hidden layer
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.activation(self.z_h)

        #computing activation(wx + b) for the output layer, we are using the
        #Softmax function here
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        """Performing the feed forward for the output"""
        # feed-forward for hidden layer
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.activation(z_h)
        #feed forward for output layer
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        """Perform the back propogation to find the errors in hidden and output
        layer, then update the weights and biases accordingly."""
        #calculating the errors
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.activation.prime(self.a_h)
        #calculating the gradients in the output layer
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)
        #calculating the gradients in the hidden layer
        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0: #adding regularization if given
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        #updating the weights and biases to minimize the cost function
        self.output_weights = self.output_weights - (self.eta*self.output_weights_gradient)
        self.output_bias = self.output_bias - (self.eta*self.output_bias_gradient)
        self.hidden_weights = self.hidden_weights - (self.eta*self.hidden_weights_gradient)
        self.hidden_bias = self.hidden_bias - (self.eta*self.hidden_bias_gradient)

    def predict(self, X):
        """Finding the prediction to see if the neural network predcited correctly."""
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        """Finding probabilities"""
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        """This is where the training happens, we are using the stochastic gradient
        descet to train the network, although this is quite a simple approach."""
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)
                #training the data in mini batches for better results
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                self.feed_forward()
                self.backpropagation()
