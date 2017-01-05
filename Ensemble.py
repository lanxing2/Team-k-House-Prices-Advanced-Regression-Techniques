
# coding: utf-8

# In[1]:

# load data
import pandas as pd
import numpy as np
from pandas import DataFrame
base_path='D:/kaggle/regression/'
all_data=DataFrame.from_csv(base_path+'filled.csv',index_col='Id')
all_data.info()


# In[2]:

all_id=all_data.index
all_y=all_data['SalePrice']
all_x=all_data.drop(['SalePrice'],axis=1)
all_x.info()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(all_x, all_y, test_size=0.2, random_state=42)


# In[15]:

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error

# A function to train several models for the train/test pair for one fold
# How to tune parameters is out of scope of this script, check out sklearn's GridSearchCV or RandomizedSearchCV
def one_fold(train_x,train_y,test_x,test_y):
    # Report shape
    print "Training set has shape: ",X_train.shape
    print "Test set has shape: ",X_test.shape

    # Random forest with mse
    print "Random forest with mse"
    rf = RandomForestRegressor(criterion='mse', n_estimators=43,max_features=0.0003,max_depth=1)
    print "Fitting random forest with mse"
    rf.fit(train_x, train_y)
    print 'Predicting on test set'
    rf_result=rf.predict(train_x)
    
    print 'Target is ',train_y
    print 'Got ',rf_result
    
    print('RFC MSE {score}'.format(score=mean_squared_error(train_y, rf_result)))
    
    # Adaboost
    print 'Adaboost linear'
    ad=AdaBoostRegressor(loss='linear',learning_rate=0.0125,n_estimators=2,random_state=42)
    print 'Fitting adaboost linear'
    ad.fit(train_x,train_y)
    print 'Predicting on test set'
    ad_result=ad.predict(train_x)
    
    print 'Target is ',train_y
    print 'Got ',ad_result
    
    print('Adaboost MSE {score}'.format(score=mean_squared_error(train_y, ad_result)))
    
    
    regressors={
        'rf':rf
    }
    
    return regressors


# In[16]:

regressors=one_fold(X_train, y_train, X_test, y_test)


# In[ ]:



