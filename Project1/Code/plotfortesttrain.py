from realparta import*
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

def bias_var_plot(Maxpolydegree, z):
    """plots a test error vs train error plot using scikits bulit in
    functionality of linear regression, this is not my own code"""
    print("Max poly degree is ", Maxpolydegree)
    print()
    X = np.zeros((len(z),Maxpolydegree)); X[:,0] = 1.0
    trials = 100
    testerror = np.zeros(Maxpolydegree)
    trainingerror = np.zeros(Maxpolydegree)
    polynomial = np.zeros(Maxpolydegree)

    for polydegree in range(1, Maxpolydegree):
        polynomial[polydegree] = polydegree
        # loop over trials in order to estimate the expectation value of the MSE
        testerror[polydegree] = 0.0
        trainingerror[polydegree] = 0.0
        for samples in range(trials):
            x_train, x_test, y_train, y_test = train_test_split(X, z, test_size=0.2)
            X_train =  scaling(x_train); X_test = scaling(x_test)
            model = LinearRegression(fit_intercept=True).fit(x_train, y_train)
            ypred = model.predict(x_train)
            ytilde = model.predict(x_test)
            testerror[polydegree] += mean_squared_error(y_test, ytilde)
            trainingerror[polydegree] += mean_squared_error(y_train, ypred)

        testerror[polydegree] /= trials
        trainingerror[polydegree] /= trials
        """print("Degree of polynomial: %3d"% polynomial[polydegree])
        print("Mean squared error on training data: %.8f" % trainingerror[polydegree])
        print("Mean squared error on test data: %.8f" % testerror[polydegree])
        print()"""

    plt.plot(polynomial, np.log10(trainingerror), label='Training Error')
    plt.plot(polynomial, np.log10(testerror), label='Test Error')
    plt.xlabel('Model complexity (Polynomial degree)')
    plt.ylabel('log10[MSE]')
    plt.title("Test and training error as a function of model complexity")
    plt.legend()
    plt.show()
    return testerror, trainingerror

n = 6 #number of samples
degree = 10 #degree of polynomial
noise = 0.01
noise_array = np.random.normal(0,0.1,n)*noise
x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
x,y = np.meshgrid(x,y)
X = create_design_matrix(x, y, degree)
z = (FrankeFunction(x, y) + noise_array).ravel()

bias_var_plot(degree, z)
