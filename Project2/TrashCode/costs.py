"""Each class represents a cost function to be minimized"""
import numpy as np

class MSE:
    """Mean Squarred Error, cost function for regression problems."""
    def __call__(self, z_predict, z):
        return np.mean((z_predict - z)**2)

    def prime(self, z_predict, z):
        return 2*(z_predict - z)/z.shape[0]

class CrossEntropy:
    """Cost function for classification problems."""
    def __call__(self, z_predict, z):
        return -np.sum( z*(1 - z_predict) - np.log(1 + np.exp(-z)))

    def prime(self, z_predict, z):
        return (z_predict - z)
