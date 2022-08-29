from realparta import *

def kfold_cross_validation(k, x, y, z, degree, reg, lambda_, imagename, name=None, plot=False):
    MSE_cv = np.zeros(degree)
    R2_cv = np.zeros(degree)
    deg = np.linspace(1,degree,degree)

    for i in range(1, degree+1):
        X = create_design_matrix(x,y,i)
        ind = np.arange(X.shape[0])
        X_train_scaled =  scaling(X);
        np.random.shuffle(ind) #randomly choose a columnindex

        X_train_scaled = X_train_scaled[ind]
        z_train = z[ind]

        X_train_scaled = np.array(np.array_split(X_train_scaled,k))#split the arrays in kfolds
        z_split = np.array(np.array_split(z_train,k))

        for j in range(0,k):
            index = np.ones(k, dtype=bool) #boolean variable
            index[j] = False
            X_test_fold = X_train_scaled[j]
            z_test_fold = z_split[j]

            X_train_fold = X_train_scaled[index]
            # have to reshape the array to match the sizes
            size = X_train_fold.shape[0]*X_train_fold.shape[1],X_train_fold.shape[2]
            X_train_fold = np.reshape(X_train_fold, size)
            z_train_fold = np.ravel(z_split[index])

            if reg == "ols":
                beta_cv = OLS(X_train_fold, z_train_fold)
                z_tilde_cv = X_test_fold @ beta_cv
            if reg == "ridge":
                beta_cv = Ridge(X_train_fold, z_train_fold, lambda_)
                z_tilde_cv = X_test_fold @ beta_cv
            if reg == "lasso":
                clf_lasso = skl.Lasso(alpha = lambda_,fit_intercept = False).fit(X_train_fold, z_train_fold)
                z_tilde_cv = clf_lasso.predict(X_test_fold)

            R2_cv[i-1] += R2(z_test_fold, z_tilde_cv)
            MSE_cv[i-1] += MSE(z_test_fold, z_tilde_cv)

        MSE_cv[i-1]/=k
        R2_cv[i-1]/=k

    if plot==True:
        if name == "terrain":
            plt.title(f"k-fold cv with k={k} and lambda={lambda_} with {reg} for terrain")
        else:
            plt.title(f"k-fold cv with k={k} and lambda={lambda_} with {reg}")
        plt.plot(deg, MSE_cv, linewidth=2, label= "MSE")
        plt.plot(deg, R2_cv, linewidth=2,  label= "R2 score")
        plt.legend()
        plt.xlabel("Complexity")
        plt.ylabel("Error")
        #plt.savefig(imagename, format='png')
        plt.show()

    return MSE_cv, R2_cv


if __name__ == '__main__':
    name = "frank"
    n = 100; degree = 15; k = 10;
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    noise = 4
    noise_array = np.random.normal(0,0.1,n)*noise
    x, y = np.meshgrid(x,y)
    z = (FrankeFunction(x, y) + noise_array).ravel()

    #imagename must have .png ending eg, lasso.png
    #kfold_cross_validation(k, x, y, z, degree, reg, lambda_, imagename, name=None, plot=False):

    #Studying cross-validation with different regression types for frank function
    #kfold_cross_validation(k, x, y, z, degree, "ols", 0, "olscvfrank.png", name, plot=True)
    #kfold_cross_validation(k, x, y, z, degree, "ridge", 0.8,"ridcvfrank.png",name, plot=True)
    kfold_cross_validation(k, x, y, z, degree, "lasso", 4,"lassofrank.png", name, plot=True)
