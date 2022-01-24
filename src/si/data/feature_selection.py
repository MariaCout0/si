import numpy as np
from scipy import stats
import warnings
from si.data import Dataset
from copy import copy

class VarianceThreshold:

    def __init__(self, threshold=0):
        """
        Variance treshold : baseline for feature selection
        :param treshold: non negative treshold value
        """
        if threshold < 0:
            warnings.warn("The treshold must be a non-negative value")
            threshold = 0
        self.threshold = threshold

    def fit(self, dataset):
        '''
        Calculate var over dataset.X
        '''
        X = dataset.X
        self._var = np.var(X, axis = 0)
        #self.F, self.p = self.score_func(dataset)

    def transform(self, dataset, inline = False):
        '''
        Deletes features with var <= threshold
        '''
        X = dataset.X
        cond = self._var > self.threshold                      
        idx = [i for i in range(len(cond)) if cond[i]]     
        X_trans = X[:,idx]                                   
        xnames = [dataset.xnames[i] for i in idx]
        if inline: 
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset 
        else:
            return Dataset(copy(X_trans), copy(dataset.Y), xnames, copy(dataset.yname))
    
    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)

class SelectKBest:

    def __init__(self, k: int, score_funcs):

        if score_funcs in (f_classification, f_regression):
            self._func = score_funcs

        if k > 0:
            self.k = k
        else:
            warnings.warn('Invalid feature number, K must be greater than 0.')

    def fit(self, dataset):
        self.F, self.P = self._func(dataset)

    def transform(self, dataset, inline=False):
        X = dataset.X
        xnames = dataset.xnames
        feat_select = sorted(np.argsort(self.F)[-self.k:])  
        x = X[:, feat_select]                               
        xnames = [xnames[feat] for feat in feat_select]

        if inline:
            dataset.X = x
            dataset.xnames = xnames
            return dataset
        else:
            return Dataset(x, copy(dataset.Y), xnames, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)


def f_classification(dataset):
    X = dataset.X
    y = dataset.Y

    args = [X[y == a, :] for a in np.unique(y)]
    F, p = stats.f_oneway(*args)
    return F, p

def f_regression(dataset):
    from scipy.stats import f

    X = dataset.X
    y = dataset.Y

    correlation_coeficient = np.array([stats.pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])
    deg_of_freedom = y.size - 2
    corr_coef_squared = correlation_coeficient ** 2
    F = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    p = f.sf(F, 1, deg_of_freedom)
    return F, p