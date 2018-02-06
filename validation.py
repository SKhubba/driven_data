from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np

class benchmark():
    # take a model, cross validate
    # return a mean log loss score, with individual breakdowns
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

        self.log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    def score(self):
        self.score = cross_val_score(self.model, self.X, self.y, scoring=self.log_loss_scorer,n_jobs=-1)
        self.mean_score = np.mean(self.score)
        return self.mean_score

