
from numpy.core.numeric import identity
from .model import Model
from ..util.metrics import mse
import numpy as np

class LinearRegression(Model):

    def __init__(self, gd = False, epochs = 1000, lr = 0.001):
        '''Linear regression Model
        epochs: number of epochs 
        lr: learning rate for GD
        '''
        super(LinearRegression,self).__init__()
        self.gd = gd
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self,dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0],1)), X))  
        self.X = X
        self.Y = Y
        self.train_gd(X, Y) if self.gd else self.train_closed(X, Y) 
        self.is_fitted = True
    
    def train_closed(self, X, Y):
        '''uses closed form linear algebra to fit the model.    # Closed form: finds the optimal value of theta without gradient
        theta=inv(XT*X)*XT*y
        '''
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        
    def train_gd(self,X,Y):               # Gradiente descend (gd): minimizes the cost function of the model                                                    
        m = X.shape[0]  
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            grad = 1/m*(X.dot(self.theta)-Y).dot(X)            
            self.theta -= self.lr*grad
            self.history[epoch] = [self.theta[:],self.cost()]
    
    def predict(self,X):
        assert self.is_fitted, "Model must be fit before predicting"
        _x = np.hstack(([1],X))
        return np.dot(self.theta, _x)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred)/2

class LinearRegressionReg(LinearRegression):

    def __init__(self, gd=False, epochs=100, lr=0.01, lbd=1):
        '''Linear regression model with L2 regularization
        :param bool gd: if True uses gradient descent (GD) to train the model
                        otherwise closed form lineal algebra. Default False.
        :param int epochs: Number of epochs for GD
        :param int lr: Learning rate for GD
        :param float lbd: lambda for the regularization'''
        super(LinearRegressionReg, self).__init__()         
        self.gd = gd
        self.theta = None
        self.epochs = epochs
        self.lr = lr 
        self.lbd = lbd            

    def train_closed(self, X, Y):
        '''Use closed form linear algebra to fit the model.
        theta=inv(XT*X+lbd*I')*XT*y'''
        n=X.shape[1]
        identity = np.eye(n)
        identity[0,0]=0
        self.theta = np.linalg.inv(X.T.dot(X)+self.lbd*identity).dot(X.T).dot(Y)
        self.is_fitted = True

    def train_gd(self, X, Y):
        '''Uses gradient descent to fit the model'''
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        lbds = np.full(m, self.lbd)
        lbds[0] = 0                                        
            grad = (X.dot(self.theta)-Y).dot(X)           
            self.theta -= (self.lr/m) * (lbds+grad)
            self.history[epoch] = [self.theta[:],self.cost()]

    def predict(self,X):
        assert self.is_fitted, "Model must be fit before predicting"
        _x = np.hstack(([1],X))
        return np.dot(self.theta, _x)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred)/2
