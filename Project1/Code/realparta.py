import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
seed = np.random.seed(10)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

def create_design_matrix(x,y,d):
    #d is the degree of the polynomial
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)

        N = len(x)
        l = int((d+1)*(d+2)/2)# Number of elements in beta (features)
        X = np.ones((N,l))

        for i in range(1, d+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,q+k] = (x**(i-k))*(y**k)

        return X

def beta_v(X, y, reg):
    """Returns the pseudo inverse of a matrix, so it can be used even if the
    matrix we are trying to invert is singular
    X is the matrix, y is the vector and reg is the type of regression"""
    if reg == "ols":
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    if beta == "ridge":
        M = X.T @ X ; size = M.shape[0]
        I = np.eye(size)
        beta = np.linalg.pinv(M + lam*I) @ X_mat.T @ y

    return beta

def scaling(data):
    #scaling the data
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_scaled[:,0] = 1 #setting first column to 1 since StandardScaler
                   #sets it to 0 leading to a singular matrix
    return data_scaled

def OLS(X_mat, y):
    """X_mat is the design matrix and y_mat is the vector"""
    # finding beta
    #print("the shape of x is ",X_mat.shape)
    #print("the shape of y is ",y.shape)

    beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y
    return beta

def Ridge(X_mat, y, lam):
    """X_mat is the design matrix and y is the vector, lam is the hyperparameter"""
    M = X_mat.T @ X_mat
    size = M.shape[0]
    I = np.eye(size)
    beta = np.linalg.inv(M + lam*I) @ X_mat.T @ y
    return beta

def Lasso(X_mat, y, lam):
    """X_mat is the design matrix and y is the vector, lam is the hyperparameter"""
    clf_lasso = skl.Lasso(alpha=lam, fit_intercept = False).fit(X_mat, y)
    ylasso = clf_lasso.predict(X_test)
    return ylasso

def predict_train(beta, x_train):
    #ytilde is for the training data
    ytilde = x_train @ beta
    return ytilde

def predict_test(beta, x_test):
    #ypredict is the prediction on the test data
    ypredict = x_test @ beta
    return ypredict

if __name__ == '__main__':

    # Making meshgrid of datapoints and compute Franke's function
    n = 5 #number of samples
    degree = 2 #degree of polynomial
    noise = 0.01
    noise_array = np.random.normal(0,0.1,n)*noise

    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    x, y = np.meshgrid(x,y)
    X = create_design_matrix(x, y, degree)
    print("the shape is x is", X.shape)

    z = (FrankeFunction(x, y) + noise_array).ravel()
    print("the shape is z is", z.shape)

    # We split the data in test and training data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2,random_state=seed)
    X_train = scaling(X_train) ; X_test = scaling (X_test)

    #beta = OLS(X_train,z_train)
    #ztilde = predict_train(beta,X_train)
    #zpredict = predict_test(beta, X_test)

    """ylasso = Lasso(X_train,z_train, 4)

    print("Lasso MSE")
    print(MSE(z_test, ylasso))
    print("Lasso R2")
    print(R2(z_test, ylasso))


    print("Training R2")
    print(R2(z_train,ztilde))
    print("Training MSE")
    print(MSE(z_train,ztilde))


    print()
    print("Test R2")
    print(R2(z_test,zpredict))
    print("Test MSE")
    print(MSE(z_test,zpredict))"""
