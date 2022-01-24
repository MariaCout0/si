from .model import Model
import numpy as np

def majority(values):
    return max(set(values, key = values.count))

def average(values):
    return sum(values)/len(values)


class Ensemble:

    def __init__(self, modelos, fvote, accuracy_score):
        '''Recebe
        modelos -  lista de modelos
        fvote - decide como a votação irá ser feita
        accuracy_score - score'''
        self.modelos = modelos               
        self.fvote = fvote
        self.accuracy_score = accuracy_score

    def fit(self, dataset):
        self.dataset = dataset
        for model in self.modelos:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self,X):
       
        assert self.is_fitted, 'Model must be fit before predicting'
        preds = [model.predict(X) for model in self.modelos]
        vote = self.fvote(preds)
        return vote

    def cost(self, X = None, y = None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        y_pred = np.ma.apply_along_axis(self.predict, axis = 0, arr=X.T)
        return self.score(y, y_pred)
