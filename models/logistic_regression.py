from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import numpy as np

np.random.seed(1)

def logistic_fit(X_train, y_train, max_iter):
    # MLPC estimation
    scaler = sk.preprocessing.StandardScaler()

    scaler.fit(X_train)  
    X_train_scaled = scaler.transform(X_train) 

    # logistic regression estimation
    param_grid = {'C': [0.1, 1, 100],
                  'l1_ratio': np.linspace(start=0, stop=1, num=2)}

    logit_cv = LogisticRegression(max_iter = max_iter, penalty="elasticnet", verbose=3, solver='saga')
    logit_cv = GridSearchCV(logit_cv, param_grid = param_grid, cv = 2, verbose=3, n_jobs=-1, scoring='roc_auc')

    logit_cv = deepcopy(logit_cv).fit(X_train_scaled, y_train)
    return logit_cv

logit_us = logistic_fit(X_train_us, y_train_us, max_iter = 10000)
logit_sm = logistic_fit(X_train_sm, y_train_sm, max_iter = 10000)
logit_os = logistic_fit(X_train_os, y_train_os, max_iter = 10000)

joblib.dump(logit_us, 'models/logit_us.sav')
joblib.dump(logit_os, 'models/logit_os.sav')
joblib.dump(logit_sm, 'models/logit_sm.sav')


