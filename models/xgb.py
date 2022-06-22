from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import joblib

def xgb_fit(X_train, y_train, n_estimator, folds, param_combs):    
    # XGB estimation:
    params = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
            }

    xgb = XGBClassifier(learning_rate=0.05, n_estimators=n_estimator, objective='binary:logistic',nthread=4, verbosity=1)

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1)
    xgb_cv = RandomizedSearchCV(xgb, param_distributions=params, scoring='roc_auc', n_iter=param_combs,
                                    n_jobs=1, cv=skf.split(X_train, y_train), verbose=2)
    # fit model no training data
    xgb = xgb_cv.fit(np.array(X_train), np.array(y_train))
    return xgb

xgb_us = xgb_fit(X_train_us, y_train_us, n_estimator=100, folds = 3, param_combs=8)
xgb_os = xgb_fit(X_train_os, y_train_os, n_estimator=100, folds = 3, param_combs=8)
xgb_sm = xgb_fit(X_train_sm, y_train_sm, n_estimator=100, folds = 3, param_combs=8)

joblib.dump(xgb_us.best_estimator_, 'models/xgb_us.sav')
joblib.dump(xgb_os.best_estimator_, 'models/xgb_os.sav')
joblib.dump(xgb_sm.best_estimator_, 'models/xgb_sm.sav')