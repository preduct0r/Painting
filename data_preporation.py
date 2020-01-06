import warnings
warnings.filterwarnings("ignore")
import sys, os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, precision_score

import matplotlib
from matplotlib import pyplot as plt


if __name__ == "__main__":
    base_dir = r'C:\Users\kotov-d\Documents\bases'

    omg_features_path = os.path.join(base_dir,'omg','feature','opensmile')
    iemocap_features_path = os.path.join(base_dir,'iemocap','feature','opensmile')
    mosei_features_path = os.path.join(base_dir,'cmu_mosei','feature', 'opensmile')


    def make_x_and_y_iemocap(x, y):
        temp = pd.concat([pd.DataFrame(np.array(x)),y], axis=1)
        temp = temp[temp['cur_label']!='xxx'][temp['cur_label']!='oth'][temp['cur_label']!='dis'][temp['cur_label']!='fru'][temp['cur_label']!='exc'] \
                                                                [temp['cur_label'] != 'sur'][temp['cur_label'] != 'fea']# [temp['cur_label'] != 'neu']
        new_x, new_y = temp.iloc[:,:-1], temp.iloc[:,-1]
        return [new_x, new_y]

    with open(os.path.join(iemocap_features_path, 'x_train.pkl'), 'rb') as f:
        iemocap_x_train = pickle.load(f)
    with open(os.path.join(iemocap_features_path, 'x_test.pkl'), 'rb') as f:
        iemocap_x_test = pickle.load(f)
    with open(os.path.join(iemocap_features_path, 'y_train.pkl'), 'rb') as f:
        iemocap_y_train = pickle.load(f).loc[:,'cur_label']
    with open(os.path.join(iemocap_features_path, 'y_test.pkl'), 'rb') as f:
        iemocap_y_test = pickle.load(f).loc[:,'cur_label']

    [iemocap_x_train, iemocap_y_train] = make_x_and_y_iemocap(iemocap_x_train, iemocap_y_train)
    [iemocap_x_test, iemocap_y_test] = make_x_and_y_iemocap(iemocap_x_test, iemocap_y_test)

    print(pd.unique(iemocap_y_train))

    def make_x_and_y_omg(x, y):
        dict_emo = {'anger': 'ang', 'happy': 'hap', 'neutral': 'neu', 'surprise': 'sur', 'disgust': 'dis', 'sad': 'sad',
                    'fear': 'fea'}
        y = y.map(lambda x: dict_emo[x])
        temp = pd.concat([pd.DataFrame(np.array(x)),y], axis=1)
        temp = temp[temp['cur_label']!='dis'][temp['cur_label'] != 'sur']\
            [temp['cur_label'] != 'fea'].reset_index(drop=True)                    # [temp['cur_label'] != 'neu']
        new_x, new_y = temp.iloc[:,:-1], temp.iloc[:,-1]
        return [new_x, new_y]

    with open(os.path.join(omg_features_path, 'x_train.pkl'), 'rb') as f:
        omg_x_train = pickle.load(f)
    with open(os.path.join(omg_features_path, 'x_test.pkl'), 'rb') as f:
        omg_x_test = pickle.load(f)
    with open(os.path.join(omg_features_path, 'y_train.pkl'), 'rb') as f:
        omg_y_train = pickle.load(f).loc[:,'cur_label']
    with open(os.path.join(omg_features_path, 'y_test.pkl'), 'rb') as f:
        omg_y_test = pickle.load(f).loc[:,'cur_label']

    [omg_x_train, omg_y_train] = make_x_and_y_omg(omg_x_train, omg_y_train)
    [omg_x_test, omg_y_test] = make_x_and_y_omg(omg_x_test, omg_y_test)

    print(pd.unique(omg_y_train))

    def make_x_and_y_mosei(x, y):
        dict_emo = {'anger': 'ang', 'happiness': 'hap', 'surprise': 'sur', 'disgust': 'dis', 'sadness': 'sad',
                    'fear': 'fea'}
        y = y.map(lambda x: dict_emo[x])
        temp = pd.concat([pd.DataFrame(np.array(x)),y], axis=1)
        temp = temp[temp['cur_label']!='dis'][temp['cur_label'] != 'sur']\
            [temp['cur_label'] != 'fea'].reset_index(drop=True)
        new_x, new_y = temp.iloc[:,:-1], temp.iloc[:,-1]
        return [new_x, new_y]

    with open(os.path.join(mosei_features_path, 'x_train.pkl'), 'rb') as f:
        mosei_x_train = pickle.load(f)
    with open(os.path.join(mosei_features_path, 'x_test.pkl'), 'rb') as f:
        mosei_x_test = pickle.load(f)
    with open(os.path.join(mosei_features_path, 'y_train.pkl'), 'rb') as f:
        mosei_y_train = pickle.load(f).loc[:,'cur_label']
    with open(os.path.join(mosei_features_path, 'y_test.pkl'), 'rb') as f:
        mosei_y_test = pickle.load(f).loc[:,'cur_label']

    [mosei_x_train, mosei_y_train] = make_x_and_y_mosei(mosei_x_train, mosei_y_train)
    [mosei_x_test, mosei_y_test] = make_x_and_y_mosei(mosei_x_test, mosei_y_test)

    print(pd.unique(mosei_y_train))








    # ==========================================================================
    # take only top 100 features
    clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
                         objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)

    clf.fit(iemocap_x_train, iemocap_y_train)

    # with open(os.path.join(r'C:\Users\kotov-d\Documents\check_relabling', 'clf' + '.pkl'), 'wb') as f:
    #     pickle.dump(clf, f, protocol=4)
    #
    # with open(os.path.join(r'C:\Users\kotov-d\Documents\check_relabling', 'clf' + '.pkl'), 'rb') as f:
    #     clf = pickle.load(f)


    dict_importance = {}
    for feature, importance in zip(range(len(clf.feature_importances_)), clf.feature_importances_):
        dict_importance[feature] = importance

    best_features = []

    for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
        if idx == 100:
            break
        best_features.append(w)

    iemocap_x_train = iemocap_x_train.loc[:,best_features]
    iemocap_x_test = iemocap_x_test.loc[:,best_features]
    omg_x_train = omg_x_train.loc[:,best_features]
    omg_x_test = omg_x_test.loc[:,best_features]
    mosei_x_train = mosei_x_train.loc[:, best_features]
    mosei_x_test = mosei_x_test.loc[:, best_features]



    temp_path = "C:\\Users\\kotov-d\\Documents\\check_relabling"


    with open(os.path.join(temp_path, 'iemocap_x_train.pkl'), 'wb') as f:
        pickle.dump(iemocap_x_train, f)
    with open(os.path.join(temp_path, 'iemocap_x_test.pkl'), 'wb') as f:
        pickle.dump(iemocap_x_test, f)
    with open(os.path.join(temp_path, 'omg_x_train.pkl'), 'wb') as f:
        pickle.dump(omg_x_train, f)
    with open(os.path.join(temp_path, 'omg_x_test.pkl'), 'wb') as f:
        pickle.dump(omg_x_test, f)
    with open(os.path.join(temp_path, 'mosei_x_train.pkl'), 'wb') as f:
        pickle.dump(mosei_x_train, f)
    with open(os.path.join(temp_path, 'mosei_x_test.pkl'), 'wb') as f:
        pickle.dump(mosei_x_test, f)


    with open(os.path.join(temp_path, 'iemocap_y_train.pkl'), 'wb') as f:
        pickle.dump(iemocap_y_train, f)
    with open(os.path.join(temp_path, 'iemocap_y_test.pkl'), 'wb') as f:
        pickle.dump(iemocap_y_test, f)
    with open(os.path.join(temp_path, 'omg_y_train.pkl'), 'wb') as f:
        pickle.dump(omg_y_train, f)
    with open(os.path.join(temp_path, 'omg_y_test.pkl'), 'wb') as f:
        pickle.dump(omg_y_test, f)
    with open(os.path.join(temp_path, 'mosei_y_train.pkl'), 'wb') as f:
        pickle.dump(mosei_y_train, f)
    with open(os.path.join(temp_path, 'mosei_y_test.pkl'), 'wb') as f:
        pickle.dump(mosei_y_test, f)


    print(iemocap_x_train.columns)
    print(omg_x_train.columns)
    print(mosei_x_train.columns)

def data_retrieval_os():
    temp_path = "C:\\Users\\kotov-d\\Documents\\check_relabling"

    with open(os.path.join(temp_path, 'iemocap_x_train.pkl'), 'rb') as f:
        iemocap_x_train = pickle.load(f)
    with open(os.path.join(temp_path, 'iemocap_x_test.pkl'), 'rb') as f:
        iemocap_x_test = pickle.load(f)
    with open(os.path.join(temp_path, 'omg_x_train.pkl'), 'rb') as f:
        omg_x_train = pickle.load(f)
    with open(os.path.join(temp_path, 'omg_x_test.pkl'), 'rb') as f:
        omg_x_test = pickle.load(f)
    with open(os.path.join(temp_path, 'mosei_x_train.pkl'), 'rb') as f:
        mosei_x_train = pickle.load(f)
    with open(os.path.join(temp_path, 'mosei_x_test.pkl'), 'rb') as f:
        mosei_x_test = pickle.load(f)


    with open(os.path.join(temp_path, 'iemocap_y_train.pkl'), 'rb') as f:
        iemocap_y_train = pickle.load(f)
    with open(os.path.join(temp_path, 'iemocap_y_test.pkl'), 'rb') as f:
        iemocap_y_test = pickle.load(f)
    with open(os.path.join(temp_path, 'omg_y_train.pkl'), 'rb') as f:
        omg_y_train = pickle.load(f)
    with open(os.path.join(temp_path, 'omg_y_test.pkl'), 'rb') as f:
        omg_y_test = pickle.load(f)
    with open(os.path.join(temp_path, 'mosei_y_train.pkl'), 'rb') as f:
        mosei_y_train = pickle.load(f)
    with open(os.path.join(temp_path, 'mosei_y_test.pkl'), 'rb') as f:
        mosei_y_test = pickle.load(f)

    return [[iemocap_x_train, iemocap_x_test, omg_x_train, omg_x_test, mosei_x_train, mosei_x_test],
        [iemocap_y_train, iemocap_y_test, omg_y_train, omg_y_test, mosei_y_train, mosei_y_test]]

def make_array(x):
    new_x = [np.array(c) for c in x]

    array_x = np.vstack(new_x)
    return array_x

def data_retrieval_openl3(option):

    temp_path = r"C:\Users\kotov-d\Documents\bases\iemocap\feature\openl3\from_yura"


    with open(os.path.join(temp_path, 'x_train.pkl'), 'rb') as f:
        iemocap_x_train =  pickle.load(f)
    with open(os.path.join(temp_path, 'x_test.pkl'), 'rb') as f:
        iemocap_x_test = pickle.load(f)
    with open(os.path.join(temp_path, 'y_train.pkl'), 'rb') as f:
        iemocap_y_train = pickle.load(f)
    with open(os.path.join(temp_path, 'y_test.pkl'), 'rb') as f:
        iemocap_y_test = pickle.load(f)

    for viborka in ['train','test']:
        if viborka=='train':
            X, Y = iemocap_x_train, iemocap_y_train
            print(type(X), '1' )
        else:
            X, Y = iemocap_x_test, iemocap_y_test
            print(type(X), "2")


        df =  pd.DataFrame(index = range(len(X)), columns=['x','y'])
        df['x'] = X
        df['y'] = Y.cur_label
        df = df[df.y!='excitement'][df.y!='excitement'][df.y!='frustration'][df.y!='other'][df.y!='unknown'][df.y!='surprise'] \
            [df.y != 'fear'][df.y != 'disgust']

        print(df.y.value_counts())


        if option == 'mean':
            mean_df = pd.DataFrame(index=range(df.shape[0]), columns=['x','y'])
            mean_df['x'] = [np.mean(c, 0).reshape(1,-1) for c in df.x]
            # print(mean_df['x'].values[3].shape)
            if viborka=='train':
                train_x = np.vstack((mean_df['x']))
                train_y = df.y
            else:
                test_x = np.vstack((mean_df['x']))
                test_y = df.y


        else:
            Xs, Ys = [], []
            for cur_x, cur_y in zip(df.x, df.y):
                for j in range(cur_x.shape[0]):
                    Xs.append(cur_x[j,:])
                    Ys.append(cur_y)

            if viborka == 'train':
                train_x = np.vstack((Xs))
                train_y = Ys
            else:
                test_x = np.vstack((Xs))
                test_y = Ys








    return [train_x, test_x, train_y, test_y]

