import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm.notebook import trange, tqdm

import lightgbm as lgb
from wandb.lightgbm import wandb_callback

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score


### USE FOR LOCAL JUPYTER NOTEBOOKS ###
DOWNLOAD_DIR = Path('../download')
DATA_DIR = Path('../data')
SUBMISSIONS_DIR = Path('../submissions')
MODEL_DIR = Path('../models')
#######################################

# WHAT NEEDS TO BE RETURNED?
def preprocess_data_lgbm():
    # Load in datasets
    X = pd.read_csv(DOWNLOAD_DIR / 'train_values.csv', index_col='building_id')
    X_test = pd.read_csv(DOWNLOAD_DIR / 'test_values.csv', index_col='building_id')
    y = pd.read_csv(DOWNLOAD_DIR / 'train_labels.csv', index_col='building_id')
    # Define categorical columns
    cat_cols = list(X.select_dtypes(include='object').columns)
    geo_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    bool_cols = [col for col in X.columns if col.startswith('has')]
    cat_cols.extend(geo_cols)
    cat_cols.extend(bool_cols)
    # Define non-categorical columns
    non_cat_cols = [x for x in X if x not in cat_cols]
    # New column ordering after the ordinal encoding (below)
    feature_names = cat_cols + non_cat_cols

    # Ordinal encode categorical vars
    t = [('ord_encoder', OrdinalEncoder(dtype=int), cat_cols)]
    ct = ColumnTransformer(transformers=t, remainder='passthrough')
    X_all_ints = ct.fit_transform(X)
    # Label encode y
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(np.ravel(y))
    
    return X_all_ints, y, cat_cols, feature_names


# Taken from the docs for lgb.train and lgb.cv
# Helpful Stackoverflow answer: 
# https://stackoverflow.com/questions/50931168/f1-score-metric-in-lightgbm
def get_ith_pred(preds, i, num_data, num_class):
    """
    preds: 1D NumPY array
        A 1D numpy array containing predicted probabilities. Has shape
        (num_data * num_class,). So, For binary classification with 
        100 rows of data in your training set, preds is shape (200,), 
        i.e. (100 * 2,).
    i: int
        The row/sample in your training data you wish to calculate
        the prediction for.
    num_data: int
        The number of rows/samples in your training data
    num_class: int
        The number of classes in your classification task.
        Must be greater than 2.
    
    
    LightGBM docs tell us that to get the probability of class 0 for 
    the 5th row of the dataset we do preds[0 * num_data + 5].
    For class 1 prediction of 7th row, do preds[1 * num_data + 7].
    
    sklearn's f1_score(y_true, y_pred) expects y_pred to be of the form
    [0, 1, 1, 1, 1, 0...] and not probabilities.
    
    This function translates preds into the form sklearn's f1_score 
    understands.
    """
    # Does not work for binary classification, preds has a different form
    # in that case
    assert num_class > 2
    
    preds_for_ith_row = [preds[class_label * num_data + i]
                        for class_label in range(num_class)]
    
    # The element with the highest probability is predicted
    return np.argmax(preds_for_ith_row)
    
def lgb_f1_micro(preds, train_data):
    y_true = train_data.get_label()
    
    num_data = len(y_true)
    num_class = 3
    
    y_pred = []
    for i in range(num_data):
        ith_pred = get_ith_pred(preds, i, num_data, num_class)
        y_pred.append(ith_pred)
    
    return 'f1', f1_score(y_true, y_pred, average='micro'), True


def make_submission_top_14_features(pipeline, title):
    """
    Given a trained pipeline object, use it to make predictions on the 
    submission test set 'test_values.csv' and write them a csv in the submissions
    folder.
    """
    # Read in test_values csv and apply data preprocessing
    # note: will create a data preprocessing pipeline or function in future
    test_values = pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')
    test_values[cat_cols] = test_values[cat_cols].astype('category')
    test_values[bool_cols] = test_values[bool_cols].astype('bool')
    test_values = pd.get_dummies(test_values)
    test_values = test_values[top_14_features]

    # Generate predictions using pipeline we pass in
    predictions = pipeline.predict(test_values)

    submission_format = pd.read_csv(DATA_DIR / 'submission_format.csv',
                                    index_col='building_id')

    my_submission = pd.DataFrame(data=predictions,
                                columns=submission_format.columns,
                                index=submission_format.index)
    
    my_submission.to_csv(SUBMISSIONS_DIR / f'{title}.csv')