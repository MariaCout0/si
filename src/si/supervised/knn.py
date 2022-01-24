
import numpy as np
from .model import Model
from si.util.util import l2_distance
from si.util.metrics import accuracy_score

class KNN():
    def __init__(self, num_n, classification = True):
        super(KNN).__init__()    # to know if the model was fitted or not
        self.k_n = num_n
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True
    
    def get_neighbors(self, x):
        distances = l2_distance(x, self.dataset.X) # calculate distances from X to all other points in the dataset 
        sorted_index = np.argsort(distances)      # sort the indices that correspond to the best distances
        return sorted_index[:self.k_n]   

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before predict"
        neighbors = self.get_neighbors(x)
        values = self.dataset.Y[neighbors].tolist() 
        # Classification
        if self.classification:
            prediction = max(set(values), key = values.count)
        else:                                       # Regression
            prediction = sum(values)/len(values)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict,axis=0,arr=self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)
