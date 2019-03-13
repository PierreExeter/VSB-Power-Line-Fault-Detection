import pandas as pd
import csv as csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import *
import statsmodels.api as sm
import gc
from sklearn.feature_selection import f_classif
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, uniform, norm
from scipy.stats import randint, poisson
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import matthews_corrcoef
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

train_meta_df = pd.read_csv("../input/metadata_train_V2.csv")
test_meta_df = pd.read_csv("../input/metadata_test_V2.csv")

X_cols = ["phase"] + train_meta_df.columns[5:].tolist()
print(train_meta_df)
print(X_cols)

# # parameters importance

# Fvals, pvals = f_classif(train_meta_df[X_cols], train_meta_df["target"])

# print("F-value | P-value | Feature Name")
# print("--------------------------------")

# for i, col in enumerate(X_cols):
    # print("%.4f"%Fvals[i]+" | "+"%.4f"%pvals[i]+" | "+col)


# # hyperparameter tuning and light GBM
# rand_seed = 135
# np.random.seed(rand_seed)

# def mcc(y_true, y_pred):
    # return matthews_corrcoef(y_true, y_pred)

# mcc_scorer = make_scorer(mcc)

# lgbm_classifier = lgbm.LGBMClassifier(
        # boosting_type='gbdt', 
        # max_depth=-1, 
        # subsample_for_bin=200000, 
        # objective="binary",
        # class_weight=None, 
        # min_split_gain=0.0, 
        # min_child_weight=0.001, 
        # subsample=1.0,
        # subsample_freq=0, 
        # random_state=rand_seed, 
        # n_jobs=1, 
        # silent=True, 
        # importance_type='split')

# param_distributions = {
    # "num_leaves": randint(16, 48),
    # "learning_rate": expon(),
    # "reg_alpha": expon(),
    # "reg_lambda": expon(),
    # "colsample_bytree": uniform(0.25, 1.0),
    # "min_child_samples": randint(10, 30),
    # "n_estimators": randint(50, 250)
# }

# clf = RandomizedSearchCV(
        # lgbm_classifier, 
        # param_distributions, 
        # n_iter=100, 
        # scoring=mcc_scorer, 
        # fit_params=None, 
        # n_jobs=1, 
        # iid=True, 
        # refit=True, 
        # cv=5, 
        # verbose=1, 
        # random_state=rand_seed, 
        # error_score=-1.0, 
        # return_train_score=True)

# clf.fit(train_meta_df[X_cols], train_meta_df["target"])

# print(clf.best_score_)
# print(clf.best_estimator_)


# xsize = 12.0
# ysize = 8.0
# fig, ax = plt.subplots()
# fig.set_size_inches(xsize, ysize)
# lgbm.plot_importance(clf.best_estimator_, ax=ax)
# plt.show()

# remove phase and median from X_cols based on the feature importance
# X_cols.remove('median')
# X_cols.remove('phase')
# print(X_cols)

# fit model
    
def classification_model(model, train_data, test_data, predictors, outcome):
    """
    make a classification model and accessing performance
    model: eg. model = LogisticRegression()
    train data: training dataframe
    test_data: test dataframe    
    predictor: list of column labels used to train the model
    outcome: column label for the objective to reach
    """

    print(model)    
    
    #Fit the model:
    print('fit he model...')
    model.fit(train_data[predictors], train_data[outcome])
  
    #Make predictions on training set:
    print('predict training set...')    
    predictions = model.predict(train_data[predictors])
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions, train_data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    # print score
    score = model.score(train_data[predictors], train_data[outcome])
    print("score: %s" % "{0:.3%}".format(score))

    #Perform k-fold cross-validation with 5 folds
    print('cross-validation...')
    # kf = cross_validation.KFold(train_data.shape[0], n_folds=5)
    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(train_data):
        # Filter training data
        train_predictors = (train_data[predictors].iloc[train,:])
    
        # The target we're using to train the algorithm.
        train_target = train_data[outcome].iloc[train]
    
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
        #Record error from each cross-validation run
        error.append(model.score(train_data[predictors].iloc[test,:], train_data[outcome].iloc[test]))
 
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    #Fit the model again so that it can be refered outside the function:
    print('fit the model for prediction...')
    model.fit(train_data[predictors], train_data[outcome]) 
    
    # predict on test set
    print('predict the model...')
    out = model.predict(test_data[predictors])
    
#    # summarize the fit of the model
#    print(metrics.classification_report(train_data[outcome], out))
#    print(metrics.confusion_matrix(train_data[outcome], out))
    
    return out


id_df = test_meta_df['signal_id']

def write_output(filename, id_df, out):
    """
    write result of machine learning to filename: 1st column is id_df and 
    2nd column is out
    """        
    out_s = pd.Series(out, name='target')
    res_df = pd.concat([id_df, out_s], axis=1)
    res_df.to_csv(filename, index=False)


print('Light GBM...')
model = LGBMClassifier(boosting_type='gbdt', class_weight=None,
        colsample_bytree=0.456616770229323, importance_type='split',
        learning_rate=0.4118819089418852, max_depth=-1,
        min_child_samples=24, min_child_weight=0.001, min_split_gain=0.0,
        n_estimators=197, n_jobs=1, num_leaves=19, objective='binary',
        random_state=135, reg_alpha=0.28274323494050946,
        reg_lambda=1.5600109217504974, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0)
out_lgbm = classification_model(model, train_meta_df, test_meta_df, X_cols, 'target')
write_output('../output/lgbm.csv', id_df, out_lgbm)

print('Logistic regression ...')
model = LogisticRegression(solver='lbfgs')
out_lr = classification_model(model, train_meta_df, test_meta_df, X_cols, 'target')
write_output('../output/lr.csv', id_df, out_lr)


print('Random Forest Classifier ...')
model = RandomForestClassifier(n_estimators=100)
out_rfc = classification_model(model, train_meta_df, test_meta_df, X_cols, 'target')
write_output('../output/rfc.csv', id_df, out_rfc)


print('SVC ...')
model = SVC(gamma='scale')
out_svc = classification_model(model, train_meta_df, test_meta_df, X_cols, 'target')
write_output('../output/svc.csv', id_df, out_svc)

print('Decision Tree Classifier ...')
model = DecisionTreeClassifier()
out_dtc = classification_model(model, train_meta_df, test_meta_df, X_cols, 'target')
write_output('../output/dtc.csv', id_df, out_dtc)

print('Gaussian NB ...')
model = GaussianNB()
out_gnb = classification_model(model, train_meta_df, test_meta_df, X_cols, 'target')
write_output('../output/gnb.csv', id_df, out_gnb)

print('KNN ...')
model = KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, metric='minkowski',
   metric_params=None, n_jobs=1, n_neighbors=19, p=2,
   weights='uniform')
out_knn = classification_model(model, train_meta_df, test_meta_df, X_cols, 'target')
write_output('../output/knn.csv', id_df, out_knn)

model = XGBClassifier(
        colsample_bylevel=0.4,
        colsample_bytree=1.0,
        gamma=0,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=5.0,
        n_estimators=100,
        reg_lambda=1.0,
        silent=False,
        subsample=1.0)
out_xgb = classification_model(model, train_meta_df, test_meta_df, X_cols, 'target')
write_output('../output/xgb.csv', id_df, out_xgb)

