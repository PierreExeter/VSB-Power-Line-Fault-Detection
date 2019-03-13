import time
from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer
import pandas as pd

# x_train, y_train, x_valid, y_valid, x_test, y_test =  # load datasets
train_meta_df = pd.read_csv("../input/metadata_train_V2.csv")
test_meta_df = pd.read_csv("../input/metadata_test_todelete.csv")

X_cols = ["phase"] + train_meta_df.columns[5:].tolist()
print(train_meta_df)
print(X_cols)


clf = xgb.XGBClassifier()
rand_seed = 135
np.random.seed(rand_seed)

def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

mcc_scorer = make_scorer(mcc)



param_grid = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}

# fit_params = {'eval_metric': 'mlogloss',
              # 'early_stopping_rounds': 10,
              # 'eval_set': [(x_valid, y_valid)]}

rs_clf = RandomizedSearchCV(
        clf, 
        param_grid, 
        n_iter=20,
        scoring=mcc_scorer,
        n_jobs=1, 
        verbose=2, 
        cv=2,
        fit_params=None, #fit_params,
        refit=False, 
        random_state=rand_seed,
        return_train_score=True)

print("Randomized search..")
search_time_start = time.time()
rs_clf.fit(train_meta_df[X_cols], train_meta_df["target"])
print("Randomized search time:", time.time() - search_time_start)

best_score = rs_clf.best_score_
best_params = rs_clf.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
# print(rs_clf.best_estimator_)


for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))
