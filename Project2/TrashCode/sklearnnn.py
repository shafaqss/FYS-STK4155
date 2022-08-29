from neededfunc import*
from sklearn.neural_network import MLPRegressor

#Regression on franke function on NN fromsklearn
n = 100 #number of samples
degree = 5 #degree of polynomial
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
z_train = z_train.ravel()
z_test = z_test.ravel()
eta_vals = np.logspace(-5, 1, 5)
lmbd_vals = np.logspace(-5, 1, 5)
epochs = 100

# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPRegressor(hidden_layer_sizes=50, activation='relu',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(X_train, z_train)

        DNN_scikit[i][j] = dnn

        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", dnn.score(X_test, z_test))
        print()

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        train_accuracy[i][j] = dnn.score(X_test, z_test)
        test_accuracy[i][j] = dnn.score(X_test, z_test)


fig, ax = plt.subplots(figsize = (10, 10))
sb.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training R2 score")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sb.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test R2 score")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
