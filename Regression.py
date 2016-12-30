
# coding: utf-8

# In[24]:

# load data
import pandas as pd
import numpy as np
from pandas import DataFrame
base_path='D:/kaggle/regression/'
all_data=DataFrame.from_csv(base_path+'train.csv',index_col='Id')
all_data.dtypes


# In[30]:

# Select all numerical
num_data=all_data.select_dtypes(include=['int64','float64'])
# MSSubClass is categorical
num_data=num_data.drop(['MSSubClass'],axis=1)
num_data.info()


# In[27]:

# Select all string
cat_data=all_data.select_dtypes(include=['object'])
cat_data.info()


# In[7]:

obj_columns=all_data.select_dtypes(include=['object'])
obj_columns.columns


# In[31]:

# Convert to categorical data
to_convert=[u'MSSubClass',u'MSZoning', u'Street', u'Alley', u'LotShape', u'LandContour',
       u'Utilities', u'LotConfig', u'LandSlope', u'Neighborhood',
       u'Condition1', u'Condition2', u'BldgType', u'HouseStyle', u'RoofStyle',
       u'RoofMatl', u'Exterior1st', u'Exterior2nd', u'MasVnrType',
       u'ExterQual', u'ExterCond', u'Foundation', u'BsmtQual', u'BsmtCond',
       u'BsmtExposure', u'BsmtFinType1', u'BsmtFinType2', u'Heating',
       u'HeatingQC', u'CentralAir', u'Electrical', u'KitchenQual',
       u'Functional', u'FireplaceQu', u'GarageType', u'GarageFinish',
       u'GarageQual', u'GarageCond', u'PavedDrive', u'PoolQC', u'Fence',
       u'MiscFeature', u'SaleType', u'SaleCondition']
print len(to_convert),' columns to convert from categorical to numerical.'


# In[32]:

for cname in to_convert:
    all_data[cname]=all_data[cname].astype('category')
cat_columns = all_data.select_dtypes(['category'])


# In[33]:

cat_columns.columns


# In[34]:

code_columns = cat_columns[cat_columns.columns].apply(lambda x: x.cat.codes)
print "Converting finished"
code_columns.info()


# In[36]:

# Convert to one-hot
count=0
one_hot_df=pd.DataFrame()
for col_name in to_convert:
    count+=1
    print 'Converting column ',col_name
    one_hot_columns=pd.get_dummies(all_data[col_name],prefix=col_name+'_')
    print count,': one-hot converted to ',type(one_hot_columns),one_hot_columns.shape
    for this_col in one_hot_columns.columns:
        one_hot_df[this_col]=one_hot_columns[this_col]
print 'All columns converted to one-hot matrix', 'count: ',count
one_hot_df.info()


# In[20]:

len(to_convert)


# In[37]:

concat_data=pd.concat([num_data, one_hot_df], axis=1)
concat_data.info()


# In[41]:

concat_data.isnan().values.sum()


# In[39]:

concat_bak=concat_data.copy(deep=True)


# In[40]:

# Replace NA
concat_bak.apply(lambda x: x.fillna(x.mean()),axis=0)
concat_bak.isnull().values.sum()


# In[67]:

concat_bak[concat_bak.isnull().any(axis=1)]


# In[47]:

# Fill Lot frontage
concat_bak[u'LotFrontage']=concat_bak[u'LotFrontage'].fillna(0)


# In[62]:

# Print columns that contains NAN
for k,v in concat_bak.isnull().any().iteritems():
    if v:
        print k


# In[50]:

# Fill masonary vesse area
concat_bak[u'MasVnrArea']=concat_bak[u'MasVnrArea'].fillna(0)


# In[66]:

# Fill garage year built
concat_bak[u'GarageYrBlt']=concat_bak[u'GarageYrBlt'].fillna(1899)


# In[68]:

concat_bak.describe()


# In[70]:

# Output 
concat_bak.to_csv(base_path+'filled.csv')


# In[ ]:



