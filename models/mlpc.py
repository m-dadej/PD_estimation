from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
import sklearn as sk
import joblib

rnd.seed(1)

def mlpc_fit(X_train, y_train, folds, n_iter):
    # MLPC estimation
    scaler = sk.preprocessing.StandardScaler()

    scaler.fit(X_train)  
    X_train_scaled = scaler.transform(X_train) 

    param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100), (100, 50, 20)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001, 0.05, 0.005],
    'learning_rate': ['constant','adaptive'],
    }

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1)
    mlpc = MLPClassifier(random_state=1, max_iter=10000, verbose = False, tol = 1e-2)
    mlpc = RandomizedSearchCV(mlpc, param_distributions = param_grid, cv = skf.split(X_train_scaled, y_train), n_iter=n_iter, verbose = 2)
    mlpc.fit(X_train_scaled, y_train)
    return mlpc


mlpc_us = mlpc_fit(X_train_us, y_train_us, folds = 3, n_iter = 5)
mlpc_os = mlpc_fit(X_train_os, y_train_os, folds = 3, n_iter = 5)
mlpc_sm = mlpc_fit(X_train_sm, y_train_sm, folds = 3, n_iter = 5)

joblib.dump(mlpc_us.best_estimator_, 'models/mlpc_us.sav')
joblib.dump(mlpc_os.best_estimator_, 'models/mlpc_os.sav')
joblib.dump(mlpc_sm.best_estimator_, 'models/mlpc_sm.sav')