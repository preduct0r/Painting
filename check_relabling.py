import sys, os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, classification_report

from data_preporation import data_retrieval_os, data_retrieval_openl3


if __name__ == "__main__":

    clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=10,
                         objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)

    # with open(os.path.join(r'C:\Users\kotov-d\Documents\check_relabling', 'clf' + '.pkl'), 'rb') as f:
    #     clf = pickle.load(f)

    df, i = pd.DataFrame(index = range(4), columns=['model', 'feature', 'num_of_labels', 'f1-score']), 0
    feature = 'openl3'

    # extract features
    if feature=='opensmile':
        [[x_train, x_test, _, _, _, _],
         [y_train, y_test, _, _, _, _]] = data_retrieval_os()
    elif feature=='openl3':
        [x_train, x_test, y_train, y_test] = data_retrieval_openl3('full')

        # y_train, y_test = y_train.loc[:,'cur_label'], y_test.loc[:,'cur_label']
    else:
        print('Some problem with feature loading')

    # make model with seed
    clf= LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
                 objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=171)


    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = round(f1_score(y_pred, y_test, average='macro'), 3)
    df.loc[i] = ['lgbm', feature, 4, f1]
    i+=1

    print(classification_report(y_pred, y_test))












