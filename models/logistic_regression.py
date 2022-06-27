from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import numpy as np

np.random.seed(1)

# logistic regression estimation
param_grid = {'C': np.logspace(-4, 2, 3),
              'l1_ratio': np.linspace(start=0, stop=1, num=3)}

logit_cv = LogisticRegression(max_iter = 100, penalty="elasticnet", solver='saga', verbose=3)
logit_cv = GridSearchCV(logit_cv, param_grid = param_grid, cv = 2, verbose=3, n_jobs=1, scoring='f1')

logit_us = logit_cv.fit(X_train_us, y_train_us)
logit_os = deepcopy(logit_cv).fit(X_train_os, y_train_os)
logit_sm = deepcopy(logit_cv).fit(X_train_sm, y_train_sm)

joblib.dump(logit_us, 'models/logit_base_us.sav')
joblib.dump(logit_os, 'models/logit_base_os.sav')
joblib.dump(logit_sm, 'models/logit_base_sm.sav')
