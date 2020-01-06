import sys, os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score

if __name__ == "__main__":
    base_dir = r'C:\Users\preductor\Documents\STC\базы'

    omg_features_path = os.path.join(base_dir,'omg','feature','opensmile')
    iemocap_features_path = os.path.join(base_dir,'iemocap','feature','opensmile')

    def make_x_and_y_iemocap(x, y):
        temp = pd.concat([pd.DataFrame(np.array(x)),y], axis=1)
        temp = temp[temp['cur_label']!='xxx'][temp['cur_label']!='oth'][temp['cur_label']!='dis'][temp['cur_label']!='exc'] \
            [temp['cur_label'] != 'sad'][temp['cur_label'] != 'hap'][temp['cur_label'] != 'sur'][temp['cur_label'] != 'fea'] \
            [temp['cur_label'] != 'fru']
        temp['cur_label'].replace('fru','ang', inplace=True)
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

    def make_x_and_y_omg(x, y):
        dict_emo = {'anger': 'ang', 'happy': 'hap', 'neutral': 'neu', 'surprise': 'sur', 'disgust': 'dis', 'sad': 'sad',
                    'fear': 'fea'}
        y = y.map(lambda x: dict_emo[x])
        temp = pd.concat([pd.DataFrame(np.array(x)),y], axis=1)
        temp = temp[temp['cur_label']!='dis'][temp['cur_label'] != 'sur'] \
            [temp['cur_label'] != 'sad'][temp['cur_label'] != 'hap'] \
            [temp['cur_label'] != 'fea']
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

    # ==========================================================================
    # take only top 100 features
    # clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
    #                      objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
    #                      subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)
    #
    # clf.fit(iemocap_x_train, iemocap_y_train)
    #
    # with open(os.path.join(r'C:\Users\preductor\PycharmProjects\STC', 'clf' + '.pkl'), 'wb') as f:
    #     clf = pickle.dump(clf, f, protocol=3)

    with open(os.path.join(r'C:\Users\preductor\PycharmProjects\STC', 'clf' + '.pkl'), 'rb') as f:
        clf = pickle.load(f)

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

    def print_results(clf, x, y):
        y_pred = clf.predict(x)
        print('f1_score = {}'.format(round(f1_score(y_pred, y, average='macro'), 3)))
        # print('accuracy = {}'.format(round(accuracy_score(y_pred, y), 3)))
        # print('recall = {}'.format(round(recall_score(y_pred, y, average='macro'), 3)))
        # print(confusion_matrix(y, y_pred.tolist()))


    clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
                         objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)
    # clf.fit(iemocap_x_train, iemocap_y_train)
    #
    #
    # print("Ground truth result on IEMOCAP test:")
    # print_results(clf, iemocap_x_test, iemocap_y_test)

    omg_shape = omg_x_train.shape[0]
    new_omg = pd.DataFrame(columns=omg_x_train.columns)
    omg_original = deepcopy(omg_x_train)

    def make_big_df(new_omg, omg_shape, clf, iemocap_x, iemocap_y, omg_x, big_df_x=None, big_df_y=None):
        if big_df_x is not None:
            clf.fit(big_df_x, big_df_y)
        else:
            clf.fit(iemocap_x, iemocap_y)
            (big_df_x, big_df_y) = (iemocap_x, iemocap_y)
        idx_to_add = []
        prediction = clf.predict(omg_x)
        probas = pd.DataFrame(index = omg_x.index,
                              data=np.hstack((clf.predict_proba(omg_x), prediction[:,np.newaxis])),
                              columns=['1_proba','2_proba','predict'])



        # отсекаем по трешхолду =====================================================
        # for idx, row in probas.iterrows():
        #     if np.max(row.iloc[:4]) > 0.40:
        #         idx_to_add.append(idx)
        # print('добавляем %s файлов из OMG на данной операции' % len(idx_to_add))

        # отсекаем по проценту от размера базы ======================================
        dict_of_max = {}
        for idx, row in probas.iterrows():
            dict_of_max[idx] = np.max(row.iloc[:2])

        idx_to_add = []
        min_max = 1.
        for idx, w in enumerate(sorted(dict_of_max, key=dict_of_max.get, reverse=True)):
            if idx == omg_shape//10:
                break
            l=dict_of_max.get(w)
            if min_max>l:
                min_max = l
            idx_to_add.append(w)
        print('минимальная уверенность: %s' % round(min_max,3))

        # ===========================================================================


        temp1 = pd.concat([big_df_x, big_df_y],axis=1)
        temp2 = omg_x.loc[idx_to_add,:].copy()
        temp2['cur_label'] = probas.predict.loc[idx_to_add]
        temp = pd.concat([temp1, temp2], axis=0)
        if new_omg is None:
            new_omg = temp2.copy()
        else:
            new_omg = pd.concat([new_omg, temp2], axis=0)
        omg_x.drop(idx_to_add, axis=0, inplace=True)
        print('осталось в OMG на текущий момент %s' % omg_x.shape[0])
        return [new_omg, temp.iloc[:,:-1], temp.iloc[:,-1]]

# =================================================================
    [big_df_x, big_df_y] = [None, None]

    for i in range(10):
        print('%d итерация\n' %(i+1))
        [new_omg, big_df_x, big_df_y] = make_big_df(new_omg, omg_shape, clf, iemocap_x_train, iemocap_y_train,
                                           omg_x_train, big_df_x, big_df_y)
        clf.fit(big_df_x, big_df_y)
        print_results(clf, iemocap_x_test, iemocap_y_test)

        temp = pd.read_csv(r'C:\Users\preductor\Documents\STC\базы\omg\meta_train.csv', delimiter=';')
        temp = temp[temp['cur_label'] != 'disgust'][temp['cur_label'] != 'surprise'][temp['cur_label'] != 'fear'] \
            [temp['cur_label'] != 'sad'][temp['cur_label'] != 'happy']

        temp['predict'] = clf.predict(omg_original)
        fin = temp[temp['predict']!='neu'].cur_name
        ff1 = []
        with open(r'C:\Users\preductor\PycharmProjects\STC\labeled_as_anger_omg_train.txt', 'w') as f:
            for i in fin:
                if i.startswith('anger'):
                    f.write(i)
                    f.write("\n")
                    ff1.append(i)

        ff2 = os.listdir(r'C:\Users\preductor\Documents\STC\базы\MEGA_TEST')
        print('на отфильтрованной выборке: точность %.3f, полнота %.3f' % \
              (round(len([w for w in ff2 if w in ff1])/float(fin.shape[0]), 3),\
              round(len([w for w in ff2 if w in ff1])/float(len(ff2)),3)))
        print(round(len([w for w in ff2 if w in ff1])))
        print('===========================================================================')

