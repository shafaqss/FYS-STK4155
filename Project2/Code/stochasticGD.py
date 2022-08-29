from neededfunc import*

def learning_schedule(t, t0=5, t1=50):
    """Implementing an adaptive learning rate for better convergence"""
    return t0/(t + t1)

def gradient(X_data, z_data, beta, bsize, lmbda=0):
    """Finding the gradient of the cost function. X_data and z_data are the data
    matrices, lambda is the regularization parameter, corresponding to Ridge
    regression, lmbda=0 corresponds to ordinary least squares regression."""
    g = (2/bsize) * X_data.T @ ((X_data @ beta) - z_data) + 2*lmbda*beta
    return g

def SGD_momentum(X_train, X_test, z_train, z_test, n_epochs, bsize, alpha=0.9, lmbda=0):
    """Implementing stochastic gradient descent with momentum. Also finds the error by
    making a prediction on the test data.
    n_epochs: the number of epochs for training
    batch_size: size of the batches, note the data must be divisible by the
                batch size, or else the training will stop.
    alpha: this is the decay factor, between 0 and 1. It is set to 0.9 by default
    lmbda: this is the regularization paramater, for ridge regression. It is set
           to 0 by default, which corresponds to OLS regression.
    """
    n = X_train.shape[0]
    if (bsize > n):
        raise Exception("Batch size cannot be greater than data size")
    if (n % bsize):
        raise Exception("Data size must be divisible by batch size")

    batches = int(n/bsize) #total number of batches
    mseepoch = np.zeros(n_epochs)
    r2epoch = np.zeros(n_epochs)
    mseepoch_analytic = np.zeros(n_epochs)
    r2epoch_analytic = np.zeros(n_epochs)

    #initialize beta randomly to start the algorithm
    beta = np.random.randn(X_train.shape[1], z_train.shape[1])

    v = 0 #momentum parameter
    ind = np.arange(n)
    for e in range(n_epochs):
        np.random.shuffle(ind)
        minibatches_X = np.split(X_train[ind], batches) #split into batches
        minibatches_z = np.split(z_train[ind], batches)

        j = 0
        for batch_X, batch_z in zip(minibatches_X, minibatches_z):
            #find the gradient for each minibatch
            grad = gradient(batch_X, batch_z, beta, bsize, lmbda)
            # Update beta with sgd momentum algorithm
            eta = learning_schedule(e*n + j)
            v = alpha*v + eta*grad
            beta = beta - v
            j = j + 1

        z_predict = X_test @ beta
        b_analytic = Ridge(X_train, z_train, lmbda)
        z_analytic_predict = X_test @ b_analytic

        mseepoch[e] = MSE(z_test, z_predict)
        r2epoch[e] = R2(z_test, z_predict)#finding mse in every epoch
        mseepoch_analytic[e] = MSE(z_test, z_analytic_predict)
        r2epoch_analytic[e] = R2(z_test, z_analytic_predict)

    return beta, mseepoch, r2epoch, mseepoch_analytic, r2epoch_analytic

#Regression on Franke function
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

# We split the data in test and training data
X_train, X_test, z_train, z_test = train_test_np(X, z)

print()
print("Info after scaling")
print("Training info")
print("Design matrix training ", X_train.shape)
print("train z ", z_train.shape)
print()
print("Test info")
print("Design matrix test ", X_test.shape)
print("test z ", z_test.shape)
print()

"""
nepoch = 100
batchsize = 20
lbd = 0.01
lbds = np.linspace(0,1,20)

b, msegd, r2gd, mseanaly, r2analy = SGD_momentum(X_train, X_test, z_train, z_test, nepoch, batchsize)
"""
epoch = np.array([50, 70, 100, 150, 200])
minibs = np.array([5, 10, 20, 50, 100])

# number of epochs vs minibatch sizes
MSE_grid_ols = np.ones((len(epoch), len(minibs)))
MSE_grid_rg = np.ones((len(epoch), len(minibs)))

for i,ep in enumerate(epoch):
    for j,mb in enumerate(minibs):
        beta = SGD_momentum(X_train, X_test, z_train, z_test, ep, mb)[0]
        z_predict = X_test @ beta
        MSE_grid_ols[i,j] = MSE(z_test, z_predict)

        beta = SGD_momentum(X_train, X_test, z_train, z_test, ep, mb, lmbda=0.07)[0]
        z_predict = X_test @ beta
        MSE_grid_rg[i,j] = MSE(z_test, z_predict)

plt.figure()
heatmap = sb.heatmap(MSE_grid_ols, annot=True,cmap="viridis",
                                      xticklabels=minibs,
                                      yticklabels=epoch,
                                      cbar_kws={'label': 'Mean squared error'},
                                      fmt = ".5")

heatmap.set_ylabel("Epochs")
heatmap.set_xlabel("Mini batch size")
heatmap.invert_yaxis()
heatmap.set_title("Epoch vs batchsize for ols")

plt.figure()
heatmap = sb.heatmap(MSE_grid_rg, annot=True,cmap="viridis",
                                      xticklabels=minibs,
                                      yticklabels=epoch,
                                      cbar_kws={'label': 'Mean squared error'},
                                      fmt = ".5")

heatmap.set_ylabel("Epochs")
heatmap.set_xlabel("Mini batch size")
heatmap.invert_yaxis()
heatmap.set_title("Epoch vs batchsize for ridge")
plt.show()
