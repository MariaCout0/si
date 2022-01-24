import numpy as np
from si.util.scale import StandardScaler

class PCA:

    def __init__(self, ncomponents = 2, using = "svd"):
        
        if ncomponents > 0 and isinstance(ncomponents, int):
            self.ncomponents = round(ncomponents)
        else:
            raise Exception("Number of components must be non negative and an integer")
        self.type = using
    
    def transform(self, dataset):
        scaled = StandardScaler().fit_transform(dataset).X.T      

        if self.type.lower()  == "svd": 
            self.u, self.s, self.vh = np.linalg.svd(scaled)
        else:
            self.cov_matrix = np.cov(scaled)                       # Covariance matrix
            # s are eigenvalues, u are eigenvectors
            self.s, self.u = np.linalg.eig(self.cov_matrix)        # Computes the eigenvalues and eigenvectors
        self.idx = np.argsort(self.s)[::-1]                        # Sorts the indexes (descending order)
        self.eigen_val =  self.s[self.idx]                         # Reorganization by index
        self.eigen_vect = self.u[:, self.idx]                      # Reorganization eigen vectors by column index

        self.sub_set_vect = self.eigen_vect[:, :self.ncomponents]  # Ordenation of vectors with principal components 
        return scaled.T.dot(self.sub_set_vect)                     

    def variance_explained(self):
        sum_ = np.sum(self.eigen_val)
        percentage = [i / sum_ * 100 for i in self.eigen_val]       # Percentage of the var explained own value / sum of own values * 100
        return np.array(percentage)
    
    def fit_transform(self, dataset):
        trans = self.transform(dataset)
        exp = self.variance_explained()
        return trans, exp
