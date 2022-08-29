"""Each class represents an activation function and its derivative is given
in the prime function.It is implemented this way for easy use in the neural network."""
import numpy as np

class Identity:
    """Used in ouput layer for regression"""
    def __call__(self, z):
        return z

    def prime(self, z):
        return np.ones(z.shape)

class LeakyRELU:
    """Here alpha is set to 0.01"""
    def __call__(self, z):
        return np.where(z<0, 0.01*z, z)

    def prime(self, z):
        return np.where(z < 0, 0.01, 1)

class RELU:
    """Simple Relu function"""
    def __call__(self, z):
        return np.maximum(z, 0)

    def prime(self, z):
        return np.where(z <= 0, 0, 1)
        #return 1.0 * (z >= 0)

class Sigmoid:
    """Used in output layer for binary calssification, also known as the
    logistic function"""
    def __call__(self, z):
        return 1/(1 + np.exp(-z))

    def prime(self, z):
        return self.__call__(z) - self.__call__(z)**2

class Softmax:
    """Used in the output layer for multiclass classification """
    def __call__(self, z):
        max_z = np.max(z, axis=1).reshape(-1, 1)
        return np.exp(z - max_z)/np.sum(np.exp(z - max_z), axis=1)[:, None]

    def prime(self, z):
        return self.__call__(z) - (self.__call__(z))**2

class Tanh:
    """Hyperbolic tan function"""
    def __call__(self, z):
        return np.tanh(z)

    def prime(self, z):
        return 1 - (self.__call__(z))**2

class Sine:
    """Hyperbolic tan function"""
    def __call__(self, z):
        return np.sin(z)

    def prime(self, z):
        return np.cos(z)

class Arctan:
    """Arctan function """
    def __call__(self, z):
        return np.arctan(z)

    def prime(self, z):
        return 1 / ((z**2) + 1)
