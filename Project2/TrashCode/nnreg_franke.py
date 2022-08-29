import activations as act
import costs as cost_function
from neededfunc import*
from NeuralNetwork_Reg import NN_Regression

#Regression on franke function
n = 100 #number of samples
degree = 6 #degree of polynomial
noise = 0.01
noise_array = np.random.normal(0,0.1,n)*noise

x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
x, y = np.meshgrid(x,y)
X = create_design_matrix(x, y, degree)
print("The shape of X is", X.shape)
z = (FrankeFunction(x, y) + noise_array).ravel()
z = z.reshape(-1,1)
print("The shape is z is", z.shape)
X_train, X_test, z_train, z_test = train_test_np(X, z)

n_inputs = X_test.shape[1]
nodes = [20, 50, 30]
n_outputs = 1 #1 category in regression case

n_epochs = 50
#eta = 0.5 #learning rate
#lmbda = 0 #regularization parameter
"""
mse_reg = np.zeros(n_epochs)
#Finding the mse with different activation function
nn1 = NN_Regression(n_inputs, nodes, n_outputs, cost_function.MSE())
nn1.create_network(act.Sigmoid(), act.Identity())
nn1.feed_forward(X_test)
print(f"MSE before training {MSE(z_test, nn1.layers[-1].a)}")

for r in range(n_epochs):
    nn1.back_propagation(X_train, z_train, eta, lmbda)
    nn1.feed_forward(X_test)
    #mse_reg[i] = MSE(z_test, NNR.layers[-1].a)

#plt.plot(np.arange(n_epochs), mse_reg)
#plt.title("MSE in Regression with Sigmoid activation func")
#plt.xlabel("Epochs")
#plt.ylabel("MSE")
#plt.show()
nn1.feed_forward(X_test)
print(f"MSE on test data after training {MSE(z_test, nn1.layers[-1].a)}")
"""

"""Finding the optimal eta and lambda"""
eta_vals = np.logspace(-5, 1, 3)
lmbd_vals = np.logspace(-5, 1, 3)
nnr = np.zeros((len(eta_vals), len(lmbd_vals)))
# grid search
for i, etaa in enumerate(eta_vals):
    for j, lmbdd in enumerate(lmbd_vals):
        eta = etaa     #learning rate
        lmbda = lmbdd  #regularization parameter

        NNR = NN_Regression(n_inputs, nodes, n_outputs, cost_function.MSE())
        NNR.create_network(act.Sigmoid(), act.Identity())
        NNR.feed_forward(X_test)
        print(f"MSE before training {MSE(z_test, NNR.layers[-1].a)}")

        for r in range(n_epochs):
            NNR.back_propagation(X_train, z_train, eta, lmbda)
            NNR.feed_forward(X_test)
            #mse_reg[i] = MSE(z_test, NNR.layers[-1].a)

        #plt.plot(np.arange(n_epochs), mse_n)
        #plt.show()

        NNR.feed_forward(X_test)
        nnr[i,j] = MSE(z_test, NNR.layers[-1].a)
        print(f"MSE on test data after training {MSE(z_test, NNR.layers[-1].a)}")

plt.figure()
heatmap = sb.heatmap(nnr, annot=True, cmap="viridis",
                                     xticklabels=eta_vals,
                                     yticklabels=lmbd_vals,
                                     cbar_kws={'label': 'Mean squared error'})
heatmap.set_ylabel("Eta (learning rate)")
heatmap.set_xlabel("Lambda")
heatmap.invert_yaxis()
heatmap.set_title("Lambda vs Eta")
plt.show()
