from .util import train_test_split
import numpy as np
import itertools

all = ['CrossValidationScore', 'GridSearchCV']
class CrossValidationScore:

    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get('cv', 3)
        self.split = kwargs.get('split', 0.8)
        self.score = kwargs.get('score', None)
        self.train_scores = None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        true_Y, pred_Y = [], []

        for _ in range(self.cv):
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)

            if not self.score:
                train_scores.append(self.model.cost())
                test_scores.append(self.model.cost(test.X, test.Y))
                pred_Y.extend(list(self.model.predict(test.X)))
            else:
                y_train = np.ma.apply_along_axis(self.model.predict, axis=1, arr=train.X)
                train_scores.append(self.score(train.Y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis=1, arr=test.X)
                test_scores.append(self.score(test.Y, y_test))
                pred_Y.extend(list(y_test))
            true_Y.extend(list(test.Y))

        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        self.true_Y = np.array(true_Y)
        self.pred_Y = np.array(pred_Y)
        return train_scores, test_scores

    def toDataframe(self):
        import pandas as pd
        return pd.DataFrame({'Train Scores': self.train_scores, 'Test Scores': self.test_scores})

class GridSearchCV:                                                 # Grid Search determines better parameters
    def __init__(self,model,dataset,parameters, **kwargs):
        self.model = model
        self.dataset = dataset
        hasparam = [hasattr(self.model, param) for param in parameters]
        if np.all(hasparam):
            self.parameters = parameters
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError ( f'Wrong parameters: {keys[index]}')
        self.kwargs= kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        for conf in itertools.product(*values):
            for i in range(len(attrs)):
                setattr(self.model, attrs[i], conf[i])
            scores = CrossValidationScore(self.model, self.dataset, **self.kwargs).run()
            self.results.append((conf, scores))
        return self.results
    
    def toDataframe(self):
        import pandas as pd
        data = dict()
        for i, k in enumerate(self.parameters.keys()):
            v = list()
            for r in self.results:
                v.append(r[0][i])
            data[k] = v
        for i in range(len(self.results[0][1][0])):
            train, test = list(), list()
            for res in self.results:
                train.append(res[1][0][i])
                test.append(res[1][1][i])
            data['Train'+str(i+1)] = train
            data['Test'+str(i+1)] = test
        return pd.DataFrame(data)
