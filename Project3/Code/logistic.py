from neededfunc import*

def cross_entropy(y, p, lmbd, theta, epsilon=1e-7):
    """Cross entropy cost function with l2 regularization, lmbd is the
    regularization parameter. Epsilon is added to avoid getting infinity
    values."""
    c = -np.mean( np.sum(y*np.log(p + epsilon), axis=1) )
    l2_reg = 1/2 * np.sum(np.square(theta[1:]))
    loss = c + lmbd*l2_reg
    return loss

def entropy_deriv(x_train, error, n_outputs, lmbd, theta):
    """Gradient of cross entropy with l2 regularization, lmbd is the
    l2 regularization prameter"""
    n = len(x_train)
    grad = 1/n * x_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), lmbd * theta[1:]]
    return grad

def softmax(z):
    """Softmax for classification, numerically stable"""
    max_z = np.max(z, axis=1).reshape(-1, 1)
    return np.exp(z - max_z)/np.sum(np.exp(z - max_z), axis=1)[:, None]

def logistic(X_train, y_train, n_epochs, bsize, lmbd, eta):
    """Implementing logistic regression for clasification with k classes.
    Uses mini-batch gradient descent to implement logistic regression, can be quite
    slow for large data sets.
    n_epochs: number of epochs for training data
    bsize: batch size
    lmbd: regularization parameter, this is l2 regulrization
    eta: learning rate
    """
    y_train = y_train.reshape(-1,1)
    n = len(X_train)
    n_inputs = X_train.shape[1]
    #n_outputs = 1
    n_outputs = y_train.shape[1]


    beta = np.random.randn(n_inputs, n_outputs)
    data_indices = np.arange(n)

    for epoch in range(n_epochs):
        for i in range(n//bsize):#n):
            chosen_datapoints = np.random.choice(data_indices, size=bsize, replace=False)
            # minibatch training data
            xi = X_train[chosen_datapoints]
            yi = y_train[chosen_datapoints]
            logits = xi.dot(beta)
            y_probab = softmax(logits)
            loss = cross_entropy(yi, y_probab, lmbd, beta)
            error = y_probab - yi
            gradients = entropy_deriv(xi, error, n_outputs, lmbd, beta)
            beta = beta - eta*gradients
    return beta

def logistic2(X_train, y_train, n_epochs, bsize, lmbd, eta):
    """Implementing logistic regression for clasification with k classes.
    This function is not reliable becasue when splitting the data, all the mnibatches
    do not have the same size, unless data size is divisible by the size of
    minibatches.
    n_epochs: number of epochs for training data
    bsize: batch size
    lmbd: regularization parameter, this is l2 regulrization
    eta: learning rate
    """
    n = len(X_train)
    n_inputs = X_train.shape[1]
    n_outputs = y_train.shape[1]
    batches = int(n/bsize) #total no of batches

    beta = np.random.randn(n_inputs, n_outputs)
    indexes = np.arange(n)

    for epoch in range(n_epochs):
        np.random.shuffle(indexes)
        minibatches_X = np.array_split(X_train[indexes], batches) #split into batches
        minibatches_y = np.array_split(y_train[indexes], batches)
        for batch_X, batch_y in zip(minibatches_X, minibatches_y):
            logits = batch_X.dot(beta)
            y_probab = softmax(logits)
            loss = cross_entropy(batch_y, y_probab, lmbd, beta)
            error = y_probab - batch_y
            gradients = entropy_deriv(batch_X, error, n_outputs, lmbd, beta)
            beta = beta - eta*gradients
    return beta

def logistic_fast(X_train, y_train, nepochs, lmbd, eta):
    """Implementing logistic regression for clasification with k classes.
    This is a faster than the function logistic(), it does not use minibatches.
    It can run fast for large data sets.
    n_iterations: number of iterations
    lmbd: regularization parameter, this is l2 regulrization
    eta: learning rate
    """
    n = len(X_train)
    n_inputs = X_train.shape[1]
    n_outputs = y_train.shape[1]

    beta = np.random.randn(n_inputs, n_outputs)

    for i in range(nepochs):
        logits = X_train.dot(beta)
        y_probab = softmax(logits)
        loss = cross_entropy(y_train, y_probab, lmbd, beta)

        error = y_probab - y_train
        gradients = entropy_deriv(X_train, error, n_outputs, lmbd, beta)
        beta = beta - eta*gradients
    return beta
