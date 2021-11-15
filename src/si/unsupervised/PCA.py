import numpy as np
from si.util.scale import StandardScaler

class PCA:

    def __init__(self, ncomponents = 2, using = "svd"):
        # ncomponents must be int
        if ncomponents > 0 and isinstance(ncomponents, int):
            self.ncomponents = round(ncomponents)
        else:
            raise Exception("Number of components must be non negative and an integer")
        self.type = using
    
    def transform(self, dataset):
        scaled = StandardScaler().fit_transform(dataset).X.T       # scale the features/standardize data

        # using numpy.linalg.svd:
        if self.type.lower()  == "svd": 
            self.u, self.s, self.vh = np.linalg.svd(scaled)
        else:
            self.cov_matrix = np.cov(scaled)                       # covariance matrix
            # s are eigenvalues, u are eigenvectors
            self.s, self.u = np.linalg.eig(self.cov_matrix)        # Compute the eigenvalues and eigenvectors
        self.idx = np.argsort(self.s)[::-1]                        # sort the indexes (descending order)
        self.eigen_val =  self.s[self.idx]                         # reorganize by index
        self.eigen_vect = self.u[:, self.idx]                      # reorganize eigen vectors by column index

        self.sub_set_vect = self.eigen_vect[:, :self.ncomponents]  # ordered vectors with principal components 
        return scaled.T.dot(self.sub_set_vect)                     # features scaled . vetores proprios ordenados


    def variance_explained(self):
        # find the explained variance   
        sum_ = np.sum(self.eigen_val)
        percentage = [i / sum_ * 100 for i in self.eigen_val]       # percentage of the var explained own value / sum of own values * 100

    def fit_transform(self, dataset):
        trans = self.transform(dataset)
        exp = self.variance_explained()
        return trans, exp