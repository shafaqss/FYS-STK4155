#part a of project 2
from neededfunc import*

def learning_schedule(t, t0=5, t1=50):
    """Implementing an adaptive learning rate for easier convergence"""
    return t0/(t + t1)

def gradient(X_data, z_data, beta, batch_size, lmbda=0):
    """Finding the gradient of the cost function. X_data and z_data are the data
    matrices, lambda is the regularization parameter, corresponding to Ridge
    regression, lmbda=0 corresponds to ordinary least squares regression."""
    g = (2/batch_size) * X_data.T @ ((X_data @ beta) - z_data) + 2*lmbda*beta
    return g

def SGD_momentum(X_train, X_test, z_train, z_test, n_epochs, batch_size, alpha=0.9, lmbda=0):
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
    #n = len(X_train)
    if (batch_size > n):
        raise Exception("Sorry, the data set cannot by divided into this\
                        many batches, try a different batch size!")
    batches = int(n/batch_size) #total number of batches
    mseforepochs = np.zeros(n_epochs)
    #initialize beta randomly to start the algorithm
    beta = np.random.randn(X_train.shape[1], 1)

    v = 0 #momentum parameter
    indexes = np.arange(n)
    for epoch in range(n_epochs):
        np.random.shuffle(indexes)
        minibatches_X = np.split(X_train[indexes], batches) #split into batches
        minibatches_z = np.split(z_train[indexes], batches)

        j = 0
        for batch_X, batch_z in zip(minibatches_X, minibatches_z):
            #find the gradient for each minibatch
            grad = gradient(batch_X, batch_z, beta, batch_size, lmbda)
            # Update beta sgd with momentum algorithm
            eta = learning_schedule(epoch*n + j)
            v = alpha*v + eta*grad
            beta = beta - v
            j = j + 1

        z_predict = X_test @ beta
        mseforepochs[epoch] = MSE(z_test, z_predict)#finding mse in every epoch
    return beta, mseforepochs

#Doing SGD on the franke function
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
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=seed)
X_train = scaling(X_train) ; X_test = scaling (X_test)
print()
print("info after scaling and stuff")
print("test info")
print("test desig matrix ",X_test.shape)
print("test z ",z_test.shape)
print()
print("training info")
print("train desig matrix ",X_train.shape)
print("X_train length is ", len(X_train))
print("train z ",z_train.shape)
print()

epochs = 50
batch_size = 50
#SGD_momentum(X_train, X_test, z_train, z_test, n_epochs, batch_size, alpha=0.9, lmbda=0)
#example of OLS(lambda=0)
beta_ols, mse_ols = SGD_momentum(X_train, X_test, z_train, z_test, epochs, batch_size)
ztilde = predict_train(beta_ols, X_train)
zpredict = predict_test(beta_ols, X_test)
print("Training MSE")
print(MSE(z_train, ztilde))
print("Test MSE")
print(MSE(z_test, zpredict))
print()

#example of Ridge with lambda = 0.3
beta_r, mse_r = SGD_momentum(X_train, X_test, z_train, z_test, epochs, batch_size, lmbda=0.3)
ztilder = predict_train(beta_r, X_train)
zpredictr = predict_test(beta_r, X_test)
print("Training MSE")
print(MSE(z_train, ztilder))
print("Test MSE")
print(MSE(z_test, zpredictr))
print()

epoch = np.array([10,50,70,100,200])
minibs = np.array([5,10,20,50,100])

# number of epochs vs minibatch sizes
MSE_grid_ols = np.ones((len(epoch), len(minibs)))
MSE_grid_rg = np.ones((len(epoch), len(minibs)))

for i,ep in enumerate(epoch):
    for j,mb in enumerate(minibs):
        beta = SGD_momentum(X_train, X_test, z_train, z_test, ep, mb)[0]
        z_predict = X_test @ beta
        MSE_grid_ols[i,j] = MSE(z_test, z_predict)

        beta = SGD_momentum(X_train, X_test, z_train, z_test, ep, mb, lmbda=0.3)[0]
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

"""plt.plot(np.arange(epochs), mse_sgd)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("SGD on Franke function")
plt.show()"""
