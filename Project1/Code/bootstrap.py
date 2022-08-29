from realparta import*
#study the bias variance trade-off

def bootstrap(B, x, y, z, reg, lambda_, imagename, degree,name=None, plot=False):
    """perform B bootstraps on a specific Regression model, here reg
    is the type pf regression, which are
      - "ols" for Ordinary least squares
      - "ridge" for ridge Regression
      - "lasso" for lasso regression """

    bias = np.zeros(degree)
    variance = np.zeros(degree)
    error = np.zeros(degree)
    deg = np.zeros(degree)

    for i in range(1,degree+1):
            X = create_design_matrix(x,y,i)
            X_train, X_test, z_train, z_test = train_test_split(X, np.reshape(z,(-1,1)), test_size=0.2)
            X_train_scaled =  scaling(X_train); X_test_scaled = scaling(X_test)
            z_pred = np.zeros((z_test.shape[0], B))

            for j in range(B):
                index = np.arange(0, len(np.ravel(z_train)), 1) #resampling
                point = resample(index)
                X_current = X_train_scaled[point,:]
                z_current = z_train[point]

                if reg == "ols":
                    beta_bo  = OLS(X_current, z_current)
                    z_pred[:,j] = np.ravel(X_test_scaled @ beta_bo)
                if reg == "ridge":
                    beta_br = Ridge(X_current, z_current, lambda_)
                    z_pred[:,j] = np.ravel(X_test_scaled @ beta_br)
                if reg == "lasso":
                    clf_lasso = skl.Lasso(alpha=lambda_,fit_intercept = False).fit(X_current, z_current)
                    z_pred[:,j] = clf_lasso.predict(X_test_scaled)

            deg[i-1] = i
            error[i-1] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
            bias[i-1] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance[i-1] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
            """print('Polynomial degree:', degree)
            print('Error:', error[i-1])
            print('Bias^2:', bias[i-1])
            print('Var:', variance[i-1])
            print('{} >= {} + {} = {}'.format(error[i-1], bias[i-1], variance[i-1], bias[i-1]+variance[i-1]))
            print()"""

    if (plot==True):
        if name == "terrain":
            plt.title(f"MSE with {reg} and lambda={lambda_} with bootstrap for terrain data")
        else:
            plt.title(f"MSE with {reg} and lambda={lambda_} with bootstrap")

        plt.plot(deg, error, label=" Error")
        plt.plot(deg, variance, label="Variance")
        plt.plot(deg, bias, label="Bias")
        plt.legend()
        plt.xlabel("Complexity(degree of polynomial)")
        plt.ylabel("MSE")
        #plt.savefig(imagename, format='png')
        plt.show()

    return error, bias, variance

if __name__ == '__main__':
    name = "Frank"
    n = 100; degree = 15; B = 10;
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    noise = 4
    noise_array = np.random.normal(0,0.1,n)*noise
    x, y = np.meshgrid(x,y)

    z = (FrankeFunction(x, y) + noise_array).ravel()

    #imagename must have .png ending eg, lasso.png
    #bootstrap(B, x, y, z, reg, lambda_, imagename, degree,name=None, plot=False):

    #Studying bootstrap with different regression types for frank function
    #bootstrap(B, x, y, z, "ols", 0,"olsfrank.png", degree, name, plot=True)
    #bootstrap(B, x, y, z, "ridge", 0.3,"ridgefrank.png", degree,name, plot=True)
    #bootstrap(B, x, y, z, "lasso", 0.4,"lassofrank.png", degree,name, plot=True)
