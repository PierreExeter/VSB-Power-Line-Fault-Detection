import pandas as pd
import csv as csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# LOAD DATA INTO DATAFRAME
train_df = pd.read_csv("input/train.csv")
test_df  = pd.read_csv("input/test.csv")

# USEFUL INFORMATION
print train_df.head(3)
print train_df.info()
print train_df.describe()

print test_df.head(3)
print test_df.info()
print test_df.describe()

# DATA CLEANUP

#drop un-insightful columns

idx = test_df['PassengerId'].values
train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test_df  = test_df.drop(['PassengerId', 'Name','Ticket','Cabin'], axis=1)

# ideas: 
# extract title from Name (Mr, Mrs,etc)
# ticket: if they boutght the same ticket, they have higher chance of survival 'ticketShared'
# combine cabins: CabinShared
# combine test and train data to have more accuracy on data cleaning

#feature by feature analysis
#Gender: replace male/female with integers
mapping = {'male':1,'female':0}
train_df['Gender'] = train_df['Sex'].map(mapping).astype(int)
test_df['Gender'] = test_df['Sex'].map(mapping).astype(int)
train_df = train_df.drop(['Sex'], axis=1)    
test_df = test_df.drop(['Sex'], axis=1)

#Embarked: fill na and replace C/Q/S with integers
mapping = {'C':0,'Q':1,'S':2}
print train_df['Embarked'].value_counts()
train_df['Embarked'] = train_df['Embarked'].fillna("S")
test_df['Embarked'] = test_df['Embarked'].fillna("S")

train_df['Embark'] = train_df['Embarked'].map(mapping).astype(int)
test_df['Embark'] = test_df['Embarked'].map(mapping).astype(int)
train_df = train_df.drop(['Embarked'], axis=1)    
test_df = test_df.drop(['Embarked'], axis=1)  

#Age: fill na with medium age
median_age = train_df['Age'].dropna().median()
train_df['Age'] = train_df['Age'].fillna(median_age)
test_df['Age'] = test_df['Age'].fillna(median_age)

#Fare: fill na with medium fare
median_fare = train_df['Fare'].dropna().median()
test_df['Fare'] = test_df['Fare'].fillna(median_fare)

print train_df.info()
print test_df.info()

# MACHINE LEARNING

def write_output(filename, idx, out):
    """
    write result of machine learning to filename: 1st column is idx and 
    2nd column is out
    """        
        
    predictions_file = open(filename, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(idx, out))
    predictions_file.close()  
    
    
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
    print 'fit he model...'
    model.fit(train_data[predictors], train_data[outcome])
  
    #Make predictions on training set:
    print 'predict training set...'    
    predictions = model.predict(train_data[predictors])
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions, train_data[outcome])
    print "Accuracy : %s" % "{0:.3%}".format(accuracy)

    # print score
    score = model.score(train_data[predictors], train_data[outcome])
    print "score: %s" % "{0:.3%}".format(score)

    #Perform k-fold cross-validation with 5 folds
    print 'cross-validation...'
    kf = KFold(train_data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (train_data[predictors].iloc[train,:])
    
        # The target we're using to train the algorithm.
        train_target = train_data[outcome].iloc[train]
    
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
        #Record error from each cross-validation run
        error.append(model.score(train_data[predictors].iloc[test,:], train_data[outcome].iloc[test]))
 
    print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

    #Fit the model again so that it can be refered outside the function:
    print 'fit the model for prediction...'    
    model.fit(train_data[predictors], train_data[outcome]) 
    
    # predict on test set
    print 'predict the model...'    
    out = model.predict(test_data[predictors])
    
#    # summarize the fit of the model
#    print(metrics.classification_report(train_data[outcome], out))
#    print(metrics.confusion_matrix(train_data[outcome], out))
    
    return out

## convert to numpy array
#train_data = train_df.values
#test_data = test_df.values
#
#X_train = train_data[:,1:]
#y_train = train_data[:,0]
#
#X_test = test_data[:,1:]

train_header = list(train_df.columns.values)
test_header = list(test_df.columns.values)



outcome_var = 'Survived'
predictor_var = test_header

model = LogisticRegression()
out_lr = classification_model(model, train_df, test_df, predictor_var, outcome_var)
write_output("output/lr.csv", idx, out_lr)

model = RandomForestClassifier(n_estimators=100)
out_rfc = classification_model(model, train_df, test_df, predictor_var, outcome_var)
write_output("output/rfc.csv", idx, out_rfc)

# create a series with feature importances !!!
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print 'feature importances: \n', featimp

# select the 3 most important variables
predictor_var = ['Fare', 'Age', 'Gender']
out_rfc2 = classification_model(model, train_df, test_df, predictor_var, outcome_var)
write_output("output/rfc2.csv", idx, out_rfc2)

predictor_var = test_header
model = SVC()
out_svc = classification_model(model, train_df, test_df, predictor_var, outcome_var)
write_output("output/svc.csv", idx, out_svc)

model = DecisionTreeClassifier()
out_dtc = classification_model(model, train_df, test_df, predictor_var, outcome_var)
write_output("output/dtc.csv", idx, out_dtc)

model = GaussianNB()
out_gnb = classification_model(model, train_df, test_df, predictor_var, outcome_var)
write_output("output/gnb.csv", idx, out_gnb)

model = KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, metric='minkowski',
   metric_params=None, n_jobs=1, n_neighbors=19, p=2,
   weights='uniform')
out_knn = classification_model(model, train_df, test_df, predictor_var, outcome_var)
write_output("output/knn.csv", idx, out_knn)

# KNN after ibea optimisation
model = KNeighborsClassifier(algorithm='kd_tree', leaf_size=11, metric='minkowski',
   metric_params=None, n_jobs=1, n_neighbors=19, p=1,
   weights='distance')
out_knn_opti = classification_model(model, train_df, test_df, predictor_var, outcome_var)
write_output("output/knn_opti.csv", idx, out_knn_opti)

####



    
