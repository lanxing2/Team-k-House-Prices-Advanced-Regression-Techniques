
# coding: utf-8

# In[3]:

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


# In[5]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(all_x, all_y, test_size=0.2, random_state=42)


# In[7]:

def reportParams(best_parameters, score):
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


# In[1]:

from sklearn.ensemble import RandomForestRegressor

print "Random forest with mse"
params={
    'n_estimators':[42,43,44],
    'max_features':[0.0002,0.0003,0.0004],
    'max_depth':[1]
        }
from sklearn.grid_search import GridSearchCV
rfc = RandomForestRegressor(criterion='mse', n_jobs=-1)
gs = GridSearchCV(rfc, params,cv=5,verbose=2)
gs.fit(X_train, y_train)
print 'Report scores'
print gs.grid_scores_
print("Report best params for random forest")
best_parameters, score, _ = min(gs.grid_scores_, key=lambda x: x[1])
reportParams(best_parameters, score)


# get the best regressor
rfc_best=gs.best_estimator_
rfc_predict=rfc.predict(X_train)
print 'Predicted values are ',rfc_predict
print('RFC MSE {score}'.format(score=mean_squared_error(y_train, rfc_predict)))


# In[ ]:




# In[56]:

from sklearn.ensemble import ExtraTreesRegressor

print "ExtraTreesRegressor"
params={
    'n_estimators':[200,300,400,500,600],
    'max_features':[0.00001,0.0001,0.0002,0.0003,0.0004],
    'max_depth':[1,2]
        }
from sklearn.grid_search import GridSearchCV
etr = ExtraTreesRegressor(criterion='mse', n_jobs=-1,random_state=42)
gs1 = GridSearchCV(etr, params,cv=5,verbose=2)
gs1.fit(X_train, y_train)
print 'Report scores'
print gs1.grid_scores_
print("Report best params for extra tree regressor")
best_parameters, score, _ = min(gs1.grid_scores_, key=lambda x: x[1])
reportParams(best_parameters, score)


# In[54]:

from sklearn.ensemble import AdaBoostRegressor

print "Adaboost with decision trees"
params={
    'n_estimators':[1,2,3],
    'learning_rate':[0.01,0.0125,0.015]
        }
from sklearn.grid_search import GridSearchCV
# Adaboost
print "Adaboost trees with linear error"
abc = AdaBoostRegressor(loss='linear',random_state=42)
gs2 = GridSearchCV(abc, params,cv=5,verbose=2)
gs2.fit(X_train, y_train)
print 'Report scores'
print gs2.grid_scores_
print("Report best params for adaboost linear")
best_parameters, score, _ = min(gs2.grid_scores_, key=lambda x: x[1])
reportParams(best_parameters, score)


# In[50]:

from sklearn.ensemble import AdaBoostRegressor

print "Adaboost with decision trees"
params={
    'n_estimators':[3],
    'learning_rate':[0.02]
        }
from sklearn.grid_search import GridSearchCV
# Adaboost
print "Adaboost trees with square error"
abc1 = AdaBoostRegressor(loss='square')
gs = GridSearchCV(abc, params,cv=5,verbose=2)
print("Report best params for adaboost square")
best_parameters, score, _ = max(gs.grid_scores_, key=lambda x: x[1])
reportParams(best_parameters, score)


# In[ ]:

from sklearn.ensemble import AdaBoostRegressor

print "Adaboost with decision trees"
params={
    'n_estimators':[3],
    'learning_rate':[0.02]
        }
from sklearn.grid_search import GridSearchCV
# Adaboost
print "Adaboost trees with exponential error"
abc1 = AdaBoostRegressor(loss='exponential')
gs = GridSearchCV(abc, params,cv=5,verbose=2)
print("Report best params for adaboost exponential")
best_parameters, score, _ = max(gs.grid_scores_, key=lambda x: x[1])
reportParams(best_parameters, score)


# In[ ]:



