from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from copy import deepcopy

# logistic regression estimation
param_grid = {'C': np.logspace(-4, 4, 4),
              'l1_ratio': np.linspace(start=0, stop=1, num=3)}

logit_cv = LogisticRegression(max_iter = 100, penalty="elasticnet", solver='saga', verbose=3)
logit_cv = GridSearchCV(logit_cv, param_grid = param_grid, cv = 3, verbose=3, n_jobs=1)

logit_us = logit_cv.fit(X_train_us, y_train_us)
logit_os = deepcopy(logit_cv).fit(X_train_os, y_train_os)
logit_sm = deepcopy(logit_cv).fit(X_train_sm, y_train_sm)


# MLPC estimation
scaler = sk.preprocessing.StandardScaler()

scaler.fit(X_train_os)  
X_train_scaled = scaler.transform(X_train_os) 

param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (100, 50, 20)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001, 0.05, 0.005],
    'learning_rate': ['constant','adaptive'],
}

folds = 3

mlpc_us = MLPClassifier(random_state=1, max_iter=10000, verbose = False, tol = 1e-2)
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1)
mlpc_us = RandomizedSearchCV(mlpc_us, param_distributions = param_grid, cv = skf.split(X_train_scaled, y_train_os), n_iter=5, verbose = 2)
mlpc_us = mlpc_us.fit(X_train_scaled, y_train_os)


# XGB estimation:
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
folds = 3
param_comb = 5

xgb = XGBClassifier(learning_rate=0.05, n_estimators=100, objective='binary:logistic',nthread=1, verbosity=1)

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1)
xgb_cv = RandomizedSearchCV(xgb, param_distributions=params, scoring='roc_auc', n_iter=param_comb,
                                   n_jobs=1, cv=skf.split(X_train_us, y_train_us), verbose=2)
# fit model no training data
xgb_us = xgb_cv.fit(np.array(X_train_us), np.array(y_train_us))



