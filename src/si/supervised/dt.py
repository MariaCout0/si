from .model import Model
from ..util.metrics import accuracy_score
import numpy as np


class Node:
    """Implementation of a simple binary tree for DT classifier."""

    def __init__(self):
        self.right = None
        self.left = None
       
        self.column = None
        self.threshold = None
      
        self.probas = None
       
        self.depth = None
        
        self.is_terminal = False


class DecisionTree(Model):

    def __init__(self, max_depth=3, min_samples_leaf=1, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        
        self.Tree = None

    def nodeProbas(self, y):
        '''
        Calculates probability of class in a given node
        '''
        probas = []
       
        for one_class in self.classes:
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)

    def gini(self, probas):
        '''Calculates gini criterion'''
        return 1 - np.sum(probas**2)

    def calcImpurity(self, y):
        '''Wrapper for the impurity calculation. Calculates probas first and then passses them
        to the Gini criterion.
        '''
        return self.gini(self.nodeProbas(y))

    def calcBestSplit(self, X, y):
        '''Calculates the best possible split for the concrete node of the tree'''

        bestSplitCol = None
        bestThresh = None
        bestInfoGain = -999

        impurityBefore = self.calcImpurity(y)

        
        for col in range(X.shape[1]):
            x_col = X[:, col]

            
            for x_i in x_col:
                threshold = x_i
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]

                if y_right.shape[0] == 0 or y_left.shape[0] == 0:
                    continue

               
                impurityRight = self.calcImpurity(y_right)
                impurityLeft = self.calcImpurity(y_left)

               
                infoGain = impurityBefore
                infoGain -= (impurityLeft * y_left.shape[0] / y.shape[0]) + \
                    (impurityRight * y_right.shape[0] / y.shape[0])

               
                if infoGain > bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold
                    bestInfoGain = infoGain

      
        if bestInfoGain == -999:
            return None, None, None, None, None, None

      

        x_col = X[:, bestSplitCol]
        x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
        y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]

        return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right

    def buildDT(self, X, y, node):
        '''
        Recursively builds decision tree from the top to bottom
        '''
        
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if X.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return

        
        splitCol, thresh, x_left, y_left, x_right, y_right = self.calcBestSplit(X, y)

        if splitCol is None:
            node.is_terminal = True

        if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            return

        node.column = splitCol
        node.threshold = thresh

        
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probas = self.nodeProbas(y_left)

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.probas = self.nodeProbas(y_right)

        
        self.buildDT(x_right, y_right, node.right)
        self.buildDT(x_left, y_left, node.left)

    def fit(self, dataset):
        self.dataset = dataset
        X, Y = dataset.getXy()
       
        self.classes = np.unique(Y)
      
        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.probas = self.nodeProbas(Y)
        self.buildDT(X, Y, self.Tree)
        self.is_fitted = True

    def predictSample(self, x, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''
        assert self.is_fitted, 'Model must be fit before predicting'
        
        if node.is_terminal:
            return node.probas

        if x[node.column] > node.threshold:
            probas = self.predictSample(x, node.right)
        else:
            probas = self.predictSample(x, node.left)
        return probas

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        pred = np.argmax(self.predictSample(x, self.Tree))
        return pred

    def cost(self, X=None, Y=None):
        X = X if X is not None else self.dataset.X
        Y = Y if Y is not None else self.dataset.Y

        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        return accuracy_score(Y, y_pred)