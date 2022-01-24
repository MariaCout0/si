
import numpy as np
import pandas as pd


def accuracy_score(y_true, y_pred):
    '''Class performance metric that computes the accuracy and y_pred'''
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1                                  # count the correct predictions
    accuracy = correct/len(y_true)
    return accuracy

def mse(y_true, y_pred, squared = True):
    '''Mean squared error regression loss funcion.
    Parameters
    
    :param numpy.array y_true: array-like of shape(n_samples,)
        Ground truth (correct) target values
    :param numpy.array y_pred: array-like of shape(n_samples,)
        Estimated target values
    _param bool squared: If True resturns MSE, if false returns RMSE'''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true-y_pred)**2, axis = 0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)
    
def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


def cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred)).sum()


def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true

def r2_score(y_true, y_pred):
    """
    R^2 regression score function.
        R^2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.
    :param numpy.array y_true : array-like of shape (n_samples,) Ground truth (correct) target values.
    :param numpy.array y_pred : array-like of shape (n_samples,) Estimated target values.
    :returns: score (float) R^2 score.
    """
    # Residual sum of squares.
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    # Total sum of squares.
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    # R^2.
    score = 1 - numerator / denominator
    return score



def mse(y_true, y_pred):
    """
    Mean squared error regression loss function.
    Parameters
    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: loss (float) A non-negative floating point value (the best value is 0.0).
    """
    return np.mean(np.power(y_true-y_pred, 2))


class ConfusionMatrix:

    def __init__(self, true_y, predict_y):
        '''Confusion Matrix implementation for the
        evaluation of the performance model by comparing the true vs the predicted values.'''
        self.true_y = np.array(true_y)
        self.predict_y = np.array(predict_y)
        self.conf_matrix = None

    def build_matrix(self):
        self.conf_matrix = pd.crosstab(self.true_y,self.predict_y, rownames = ["True values"], colnames = ["Predicted values"])
    
    def toDataframe(self):
        return pd.DtaFrame(self.build_matrix())