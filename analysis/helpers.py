import pandas as pd

def load_jupyter_essentials():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import pickle
    import lightgbm as lgb

    from pathlib import Path
    from lightgbm import LGBMClassifier
    from pprint import pprint

    from sklearn.metrics import mean_squared_error, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV 
    from sklearn.model_selection import cross_val_score, StratifiedKFold



    ############ USE FOR GOOGLE COLAB ############
    # DATA_DIR = Path('/content/drive/MyDrive/Work/Delivery/Current/Earthquake_damage/data')
    # SUBMISSIONS_DIR = Path('drive/MyDrive/Work/Delivery/Current/Earthquake_damage/submissions')
    # MODEL_DIR = Path('/content/drive/MyDrive/Work/Delivery/Current/Earthquake_damage/models')

    # from google.colab import drive
    # drive.mount('/content/drive')
    #############################################


    ### USE FOR LOCAL JUPYTER NOTEBOOKS ###
    DATA_DIR = Path('../download')
    SUBMISSIONS_DIR = Path('../submissions')
    MODEL_DIR = Path('../models')
    #######################################

    # The code runs the same if working on Jupyter or Colab, just need to change the 
    # dirs above

    X = pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id')

    categorical_columns = X.select_dtypes(include='object').columns
    bool_columns = [col for col in X.columns if col.startswith('has')]
    X[categorical_columns] = X[categorical_columns].astype('category')
    X[bool_columns] = X[bool_columns].astype('bool')

    X = pd.get_dummies(X)
    y = pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')

def make_submission_top_14_features(pipeline, title):
    """
    Given a trained pipeline object, use it to make predictions on the 
    submission test set 'test_values.csv' and write them a csv in the submissions
    folder.
    """
    # Read in test_values csv and apply data preprocessing
    # note: will create a data preprocessing pipeline or function in future
    test_values = pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')
    test_values[categorical_columns] = test_values[categorical_columns].astype('category')
    test_values[bool_columns] = test_values[bool_columns].astype('bool')
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