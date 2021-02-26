#!/usr/bin/env python
# coding: utf-8

# # Using LightGBM as designed (not through sklearn API)
# 
# ## Automatically Encode Categorical Columns
# 
# I've been encoding the geo_level columns as numeric this whole time. Can it perform better by using categorical columns?
# 
# LGBM can handle categorical features directly. No need to OHE them. But they must be ints. 
# 
# 1. Load in X
# 2. Label Encode all the categorical features
#  - All `object` dypes are categorical and need to be LabelEncoded

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from pathlib import Path

### USE FOR LOCAL JUPYTER NOTEBOOKS ###
DATA_DIR = Path('../download')
SUBMISSIONS_DIR = Path('../submissions')
MODEL_DIR = Path('../models')
#######################################

X = pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id')
categorical_columns = X.select_dtypes(include='object').columns
bool_columns = [col for col in X.columns if col.startswith('has')]

X_test = pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')
y = pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')


# In[2]:


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

label_enc = LabelEncoder()

t = [('ord_encoder', OrdinalEncoder(dtype=int), categorical_columns)]
ct = ColumnTransformer(transformers=t, remainder='passthrough')


# In[5]:


X_all_ints = ct.fit_transform(X)
y = label_enc.fit_transform(y.values)


# In[9]:


# Note that append for pandas objects works differently to append with
# python objects e.g. python append modifes the list in-place
# pandas append returns a new object, leaving the original unmodified
not_categorical_columns = X.select_dtypes(exclude='object').columns
cols_ordered_after_ordinal_encoding = categorical_columns.append(not_categorical_columns)


# In[10]:


cols_ordered_after_ordinal_encoding


# In[11]:


geo_cols = pd.Index(['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'])
cat_cols_plus_geo = categorical_columns.append(geo_cols)


# In[36]:


list(cat_cols_plus_geo)


# In[12]:


train_data = lgb.Dataset(X_all_ints, label=y, feature_name=list(cols_ordered_after_ordinal_encoding),
                        categorical_feature=list(cat_cols_plus_geo))

# train_data = lgb.Dataset(X_all_ints, label=y)


# In[13]:


validation_data = lgb.Dataset('validation.svm', reference=train_data)


# In[31]:


# Taken from the docs for lgb.train and lgb.cv
# Helpful Stackoverflow answer: 
# https://stackoverflow.com/questions/50931168/f1-score-metric-in-lightgbm
from sklearn.metrics import f1_score

# what is preds? Is each element an int (0, 1, 2)?
# or is a list of probabilities [0.12, 0.18, 0.7]?
def lgb_f1_micro(preds, train_data):
    y_true = train_data.get_label()
#     y_pred = [np.argmax(p) for p in preds]
    y_pred = np.round(preds)
    return 'f1', f1_score(y_true, y_pred, average='micro'), True


# In[28]:


probs = [[.12, 0.18, 0.7],
         [0.2, 0.5, 0.3]]
[np.argmax(p) for p in probs]


# In[14]:


param = {'num_leaves': 120,
#          'num_iterations': 240,
         'min_child_samples': 40,
         'learning_rate': 0.2,
         'boosting_type': 'goss',
         'objective': 'multiclass',
         'num_class': 3}


# In[32]:


# LGBM seem to hate using plurals. Why???
num_round = 10
lgb.cv(param, train_data, num_round, nfold=5, 
       categorical_feature=list(cat_cols_plus_geo),
       feval=lgb_f1_micro)


# In[ ]:




