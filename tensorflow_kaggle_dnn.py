
# coding: utf-8

# In[21]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


# In[3]:

#Divide columns
CONTINUOUS_COLUMNS = ["LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","ExterQual","ExterCond",
	"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","HeatingQC",
	"1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr",
	"KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageYrBlt","GarageFinish","GarageCars",
	"GarageArea","GarageQual","GarageCond","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC",
	"MiscVal","MoSold","YrSold"]
CATEGORICAL_COLUMNS = ["MSSubClass","MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood",
	"Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType",
	"Foundation","Heating","CentralAir","Electrical","GarageType","PavedDrive","Fence","MiscFeature","SaleType","SaleCondition"]
LABEL_COLUMN = "SalePrice"
COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + [LABEL_COLUMN]
print("Continuous columns {0:2d}".format(len(CONTINUOUS_COLUMNS)))
print("Categorical columns {0:2d}".format(len(CATEGORICAL_COLUMNS)))
print("Total useful columns {0:2d}".format(len(COLUMNS)))


# In[112]:

def input_clean(raw_data):
    data = raw_data.drop('Id', 1)
    data = data.replace({                            
                            'MSSubClass':{
                                            20:'class20',
                                            30:'class30',
                                            40:'class40',
                                            45:'class45',
                                            50:'class50',
                                            60:'class60',
                                            70:'class70',
                                            75:'class75',
                                            80:'class80',
                                            85:'class85',
                                            90:'class90',
                                            120:'class120',
                                            150:'class150',
                                            160:'class160',
                                            180:'class180',
                                            190:'class190'
                                            },
                            'MSZoning': {'C (all)': 'C'
                                            },
                            'ExterQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1
                                            },
                            'ExterCond': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1
                                            },
                            'BsmtQual':{ 
                                            'Ex':5,
                                            'Gd':4,
                                            'TA':3,
                                            'Fa':2,
                                            'Po':1,
                                            'NoBsmt': 0},
                            'BsmtCond': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoBsmt': 0},
                            'BsmtExposure': {'Gd':4,
                                            'Av':3,
                                            'Mn':2,
                                            'No':1,
                                            'NoBsmt':0},
                            'BsmtFinType1':{'GLQ':6,
                                            'ALQ':5,
                                            'BLQ':4,
                                            'Rec':3,
                                            'LwQ':2,
                                            'Unf':1,
                                            'NoBsmt':0},
                            'BsmtFinType2':{
                                            'GLQ':6,
                                            'ALQ':5,
                                            'BLQ':4,
                                            'Rec':3,
                                            'LwQ':2,
                                            'Unf':1,
                                            'NoBsmt':0},
                            'HeatingQC': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1
                                            },
                            'KitchenQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1},
                            'Functional': {'Typ': 7,
                                            'Min1': 6,
                                            'Min2': 5,
                                            'Mod': 4,
                                            'Maj1': 3,
                                            'Maj2': 2,
                                            'Sev': 1,
                                            'Sal': 0},
                            'FireplaceQu': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoFireplace': 0 
                                            },
                            'GarageFinish': {
                                            'Fin':3,
                                            'RFn':2,
                                            'Unf':1,
                                            'NoGarage':0
                                             },
                            'GarageQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoGarage': 0},
                            'GarageCond': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoGarage': 0},
                            'PoolQC': {'Ex':4,
                                            'Gd':3,
                                            'TA':2,
                                            'Fa':1,
                                            'NoPool':0
                                       }
                            })
    #fill NaN
    #CONTINUOUS_COLUMNS
    data['LotFrontage']=data['LotFrontage'].fillna(0)
    data['MasVnrArea']=data['MasVnrArea'].fillna(0)
    data['GarageYrBlt']=data['GarageYrBlt'].fillna(1899)
    
    #CONTINUOUS_COLUMNS==>CATEGORICAL_COLUMNS
    data['BsmtQual']=data['BsmtQual'].fillna(0)
    data['BsmtCond']=data['BsmtCond'].fillna(0)
    data['BsmtExposure']=data['BsmtExposure'].fillna(0)
    data['BsmtFinType1']=data['BsmtFinType1'].fillna(0)
    data['BsmtFinType2']=data['BsmtFinType1'].fillna(0)
    data['FireplaceQu']=data['FireplaceQu'].fillna(0)
    data['GarageFinish']=data['GarageFinish'].fillna(0)
    data['GarageQual']=data['GarageQual'].fillna(0)
    data['GarageCond']=data['GarageCond'].fillna(0)
    data['PoolQC']=data['PoolQC'].fillna(0)
    
    #CATEGORICAL_COLUMNS
    data['Alley']=data['Alley'].fillna('NoAlley')
    data['MasVnrType']=data['MasVnrType'].fillna('NoMasVnr')
    data['GarageType']=data['GarageType'].fillna('NoGarage')
    data['Electrical']=data['Electrical'].fillna('NoElec')
    data['Fence']=data['Fence'].fillna('NoFc')
    data['MiscFeature']=data['MiscFeature'].fillna('NoFtr')
    
    return data


# In[110]:

#Check if the COLUMNS is match with the original data
folder = '/Users/lanxing/Desktop/Machine Learning/kaggle/'
check_columns = pd.read_csv(folder+"train.csv").columns.tolist()
check_dict = {}
for l in check_columns:
    check_dict[l] = 1
for l in COLUMNS:
    if l in check_dict:
        del check_dict[l]
    else:
        check_dict[l] = -1
print(check_dict)


# In[114]:

folder = '/Users/lanxing/Desktop/Machine Learning/kaggle/'
raw_train = pd.read_csv(folder+"train.csv")
print(raw_train.shape)
i=0
while i<raw_train.describe().shape[1]:
    print(raw_train.describe().iloc[:,i:min(i+9,raw_train.describe().shape[1])])
    i=i+10


# In[115]:

#Describe the training data after cleaning
train = input_clean(raw_train)
print(train.describe().shape[1])
#print(train.iloc[36:45,21:40])
i=0
while i<train.describe().shape[1]:
    print(train.describe().iloc[:,i:min(i+9,train.describe().shape[1])])
    i=i+10


# In[9]:

#Check the cleaning data
check_columns = train.describe().columns.tolist()
check_dict = {}
for l in check_columns:
    check_dict[l] = 1
for l in CONTINUOUS_COLUMNS:
    if l in check_dict:
        del check_dict[l]
    else:
        check_dict[l] = -1
print(check_dict)
for l in train.columns.tolist():
    if train.loc[:,l].isnull().values.any():
        print(l)


# In[35]:

input_stat = train.describe().loc[['min','max'],:]
input_stat = input_stat.drop('MSSubClass', 1)
max_price = input_stat.loc['max','SalePrice'].tolist()
min_price = input_stat.loc['min','SalePrice'].tolist()
input_stat.to_csv(folder+"input_stat.csv")
#print(type(max_price),type(min_price))


# In[102]:

def input_pdtotf(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    #feature_cols = dict(continuous_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label



# In[119]:

#categorical_options = {}
#for l in CATEGORICAL_COLUMNS:
#    categorical_options[l]=train.loc[:,l].value_counts().index.values.tolist()
#print(categorical_options)
CATEGORICAL_OPTIONS = {'MasVnrType': ['None', 'BrkFace', 'Stone', 'BrkCmn', 'NoMasVnr'], 
                       'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], 
                       'Exterior1st': ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood', 'CemntBd', 'BrkFace', 'WdShing', 'Stucco', 'AsbShng', 'Stone', 'BrkComm', 'AsphShn', 'ImStucc', 'CBlock'], 
                       'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', 'NoElec'], 
                       'HouseStyle': ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin'], 
                       'Foundation': ['PConc', 'CBlock', 'BrkTil', 'Slab', 'Stone', 'Wood'], 
                       'GarageType': ['Attchd', 'Detchd', 'BuiltIn', 'NoGarage', 'Basment', 'CarPort', '2Types'], 
                       'RoofStyle': ['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'], 
                       'CentralAir': ['Y', 'N'], 
                       'Utilities': ['AllPub', 'NoSeWa'], 
                       'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'], 
                       'MiscFeature': ['NoFtr', 'Shed', 'Othr', 'Gar2', 'TenC'], 
                       'PavedDrive': ['Y', 'N', 'P'], 
                       'LandSlope': ['Gtl', 'Mod', 'Sev'], 
                       'SaleType': ['WD', 'New', 'COD', 'ConLD', 'ConLw', 'ConLI', 'CWD', 'Oth', 'Con'], 
                       'Heating': ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], 
                       'Exterior2nd': ['VinylSd', 'MetalSd', 'HdBoard', 'Wd Sdng', 'Plywood', 'CmentBd', 'Wd Shng', 'Stucco', 'BrkFace', 'AsbShng', 'ImStucc', 'Brk Cmn', 'Stone', 'AsphShn', 'Other', 'CBlock'], 
                       'MSSubClass': ['class20', 'class60', 'class50', 'class120', 'class30', 'class160', 'class70', 'class80', 'class90', 'class190', 'class85', 'class75', 'class45', 'class180', 'class40'], 
                       'Condition2': ['Norm', 'Feedr', 'Artery', 'RRNn', 'PosN', 'RRAn', 'RRAe', 'PosA'], 
                       'Condition1': ['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'], 
                       'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'], 
                       'RoofMatl': ['CompShg', 'Tar&Grv', 'WdShngl', 'WdShake', 'Membran', 'Metal', 'ClyTile', 'Roll'], 
                       'Neighborhood': ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber', 'IDOTRR', 'ClearCr', 'SWISU', 'StoneBr', 'MeadowV', 'Blmngtn', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'], 
                       'Fence': ['NoFc', 'MnPrv', 'GdPrv', 'GdWo', 'MnWw'], 
                       'SaleCondition': ['Normal', 'Partial', 'Abnorml', 'Family', 'Alloca', 'AdjLand'], 
                       'BldgType': ['1Fam', 'TwnhsE', 'Duplex', 'Twnhs', '2fmCon'], 
                       'Alley': ['NoAlley', 'Grvl', 'Pave'], 
                       'Street': ['Pave', 'Grvl'], 
                       'MSZoning': ['RL', 'RM', 'FV', 'RH', 'C']}


# In[125]:

#Defining FeatureColumns and Creating the Regressor
# Creates features column from each continuous feature column name (k) to
continuous_feature = [tf.contrib.layers.real_valued_column(k)
                  for k in CONTINUOUS_COLUMNS]
# Creates features column from each categorical feature column name (k)
#categorical_feature = [tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_keys(
#                      column_name=k, keys=CATEGORICAL_OPTIONS[k]),dimension=8) for k in CATEGORICAL_COLUMNS]
categorical_feature = [tf.contrib.layers.one_hot_column(tf.contrib.layers.sparse_column_with_keys(
                      column_name=k, keys=CATEGORICAL_OPTIONS[k])) for k in CATEGORICAL_COLUMNS]
#categorical_feature = [tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket(k, hash_bucket_size=30),dimension=30) for k in CATEGORICAL_COLUMNS]


feature_cols = continuous_feature + categorical_feature
regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=feature_cols, hidden_units=[100, 100])


# In[126]:

#Training the Regressor
regressor.fit(input_fn=lambda: input_pdtotf(train), steps=1000)


# In[ ]:



