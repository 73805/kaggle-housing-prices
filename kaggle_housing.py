import pandas as pd
import random
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import csv

# Rounding function
def roundNear(x, base):
    return int(base * round(float(x)/base))

# Read in the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Drop some outliers
train.drop(train[train["GrLivArea"] > 4000].index, inplace=True)
train.drop(train[train["LotArea"] > 100000].index, inplace=True)
train.drop(train[train["BsmtFinSF1"] > 5500].index, inplace=True)

target = train.pop('SalePrice').values

# Engineering some features (pre unskewing)
test['NetBath'] = test['FullBath'] + test['BsmtFullBath'] + test['HalfBath'] + test['BsmtHalfBath']
train['NetBath'] = train['FullBath'] + train['BsmtFullBath'] + train['HalfBath'] + train['BsmtHalfBath']
test['TotalGrSF'] = test['1stFlrSF'] + test['2ndFlrSF'] - test['LowQualFinSF']
train['TotalGrSF'] = train['1stFlrSF'] + train['2ndFlrSF'] - train['LowQualFinSF']
test['NetPorchSF'] = test['WoodDeckSF'] + test['OpenPorchSF'] + test['ScreenPorch']
train['NetPorchSF'] = train['WoodDeckSF'] + train['OpenPorchSF'] + train['ScreenPorch']
test['YearsSinceRemodel'] = test['YrSold'] - test['YearRemodAdd']
train['YearsSinceRemodel'] = train['YrSold'] - train['YearRemodAdd']
test['YearsSinceGarage'] = test['YrSold'] - test['GarageYrBlt']
train['YearsSinceGarage'] = train['YrSold'] - train['GarageYrBlt']

# Defining rating mappings
standard_dict = {'Ex': 1.0, 'Gd': .8, 'TA': .6, 'Fa': .4, 'Po': .2, 'NA': 0}
alternate_dict = {'Y': 1.0, 'P': 0.5, 'N': 0, 'NA': 0}
garage_qual = {'Ex': .8, 'Gd': .8, 'TA': .6, 'Fa': .4, 'Po': .2, 'NA': 0}
garage_dict = {'Fin': 1.0, 'RFn': .66, 'Unf': .33, 'NA': 0}
basement_dict = {'Gd': 1.0, 'Av': 0.75, 'Mn': 0.5, 'No': .25, 'NA': 0}
basement_fin_dict = {'GLQ': 1.0, 'ALQ': .85, 'BLQ': .7, 'Rec': .7, 'LwQ': .6, 'Unf': .2, 'NA': 0}
# Applying the Standard mapping
standard_dict_feats = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                       'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageCond']
for feat in standard_dict_feats:
    train[feat] = train[feat].fillna('NA').map(standard_dict)
for feat in standard_dict_feats:
    test[feat] = test[feat].fillna('NA').map(standard_dict)

# Applying non-standard Mappings
train['CentralAir'] = train['CentralAir'].map(alternate_dict)
test['CentralAir'] = test['CentralAir'].map(alternate_dict)
train['PavedDrive'] = train['PavedDrive'].fillna('NA').map(alternate_dict)
test['PavedDrive'] = test['PavedDrive'].fillna('NA').map(alternate_dict)
train['GarageQual'] = train['GarageQual'].fillna('NA').map(garage_qual)
test['GarageQual'] = test['GarageQual'].fillna('NA').map(garage_qual)
train['GarageFinish'] = train['GarageFinish'].fillna('NA').map(garage_dict)
test['GarageFinish'] = test['GarageFinish'].fillna('NA').map(garage_dict)
train['BsmtExposure'] = train['BsmtExposure'].fillna('NA').map(basement_dict)
test['BsmtExposure'] = test['BsmtExposure'].fillna('NA').map(basement_dict)
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('NA').map(basement_fin_dict) + train['BsmtFinType2'].fillna('NA').map(basement_fin_dict)
test['BsmtFinType1'] = test['BsmtFinType1'].fillna('NA').map(basement_fin_dict) + test['BsmtFinType2'].fillna('NA').map(basement_fin_dict)
train['ExterScore'] = (train['ExterQual'] + train['ExterCond']) / 2
test['ExterScore'] = (test['ExterQual'] + test['ExterCond']) / 2
train['TotalBsmtSF'] = train['TotalBsmtSF'].fillna(0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
train['BsmtScore'] = ((train['BsmtQual'] + train['BsmtCond']) / 2) * (train['TotalBsmtSF'])
test['BsmtScore'] = test['BsmtQual'] * test['TotalBsmtSF']

# Cobining Train and Test into a single Dataframe
df = train.append(test, ignore_index=True)

# Convert MSSubClass for 1/0 encoding
df['MSSubClass'] = df['MSSubClass'].astype(str)
# Preserve an unscaled GrLivArea
true_gr = df['GrLivArea']
df['YearBuilt'] = df['YearBuilt'] - min(df['YearBuilt'])
df.drop('Id', axis=1, inplace=True)

# Unskewing Features
numeric_feats = df.dtypes[df.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.7]
skewed_feats = skewed_feats.index
df[skewed_feats] = np.log1p(df[skewed_feats])

df['TrueLiv'] = true_gr
# 0/1 encoding categorical features
df = pd.get_dummies(df)

# filling NA's with the mean of the column (technically data snooping)
df = df.fillna(df.mean())

# Dropping the 0/1 encoded categories absent in official test data
drop_cols = ['Utilities_NoSeWa', 'Condition2_RRAe',
'Condition2_RRAn', 'Condition2_RRNn', 'HouseStyle_2.5Fin', 'RoofMatl_Membran', 
'RoofMatl_Metal', 'RoofMatl_Roll', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 
'Exterior2nd_Other', 'Heating_Floor', 'Heating_OthW', 'Electrical_Mix', 'PoolQC_Fa', 'MiscFeature_TenC']
df.drop(drop_cols, axis=1, inplace=True)

############################################################

# Splitting Data and Applying Algorithms

############################################################

# The training portion of df
data = df[:len(train)]

# Boolean to indicate testing on training subsets or producing kaggle submission
create_submission = True

if not create_submission:
    testSize = 0.3
else:
    testSize = 0.0

# Create Testing set either by splitting training data or using kaggle testing data
rand = random.randint(1, 100)
X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                    test_size=testSize,
                                                    random_state=rand)

if create_submission:
    X_test = df.tail(len(test))

# Experimental:
# Pre-classifying fancy houses in the test set to boost expensive property indicators
X_train['Fancy'] = (np.sign(y_train - 500000) + 1) /2
fnc_target = X_train['Fancy']
fnc_data = X_train.drop('Fancy', axis=1)
fnc_test = X_test
fnc = xgb.XGBRegressor(colsample_bytree=0.3, gamma=0.0, learning_rate=0.01,
                            max_depth=4, min_child_weight=1.5, n_estimators=2500,
                            reg_alpha=0.9, reg_lambda=0.6, subsample=0.2,
                            seed=rand, silent=1)
fnc.fit(fnc_data, fnc_target)
X_test['Fancy'] = fnc.predict(X_test)

# Creating and fitting the classifier 
rand = random.randint(1, 100)
clf = xgb.XGBRegressor(colsample_bytree=0.3, gamma=0.0, learning_rate=0.01,
                            max_depth=4, min_child_weight=1.5, n_estimators=2500,
                            reg_alpha=0.9, reg_lambda=0.6, subsample=0.2,
                            seed=rand, silent=1)
clf.fit(X_train, y_train)

# Subset testing
if not create_submission:
    p = clf.predict(X_test)
    q = clf.predict(X_test)
    # Rounding seems to help (targets are round numbers)
    for i in range(0, len(p)):
        p[i] = roundNear(p[i], 250)
    # Printing score
    print np.sqrt(mean_squared_error(np.log(y_test), np.log(p)))
    # Experimental:
    # Fitting a classifier on test and errors to check prominent features related to error
    abs_err = abs(y_test - p)
    err_clf = xgb.XGBRegressor()
    err_clf.fit(X_test, abs_err)
    # err_clf.feature_importances_
else:
    # Above process plus csv generation
    p = clf.predict(X_test)
    for i in range(0, len(p)):
        p[i] = roundNear(p[i], 250)
        
    ids = test['Id'].values
    print 'Producing Kaggle Submission'
    predictions_file = open('new_submission.csv', 'wb')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['Id', 'SalePrice'])
    open_file_object.writerows(zip(ids, p))
    predictions_file.close()
    print 'Done.'
