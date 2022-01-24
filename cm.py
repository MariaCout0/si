import numpy as np
import pandas as pd

class ConfusionMatrix:
    def __call__(self, true_y, pred_y):
        self.true_y = np.array(true_y)
        self.pred_y = np.array(pred_y)
        self.conf = None
        return self.toDataframe()
    
    def calc(self):
        cam = pd.crosstab(self.true_y, self.pred_y, rownames=['Actual'], colnames=['Predicted'], margins=True)
        return cam
    
    def toDataframe(self):
        return self.conf