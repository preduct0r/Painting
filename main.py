import sys, os
import random
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import pickle
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

sys.path.append('../')
from pytorch.common.datasets_parsers.av_parser import AVDBParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error

from pytorch.common.datasets_parsers.av_parser import AVDBParser
from voice_feature_extraction import OpenSMILE
from accuracy import Accuracy, Accuracy_regression
from itertools import combinations_with_replacement


def regression(X_train, X_test, y_train, y_test, label_type, pca_dim=100):
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train, X_test]))

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if label_type != 'labels':
        scaler = MinMaxScaler()
        temp1, temp2 = y_train.loc[:, ['valence', 'arousal']], y_test.loc[:, ['valence', 'arousal']]
        temp = pd.concat((temp1, temp2))
        scaler.fit(temp.values)

        y_train = scaler.transform(temp1.values)
        y_test = scaler.transform(temp2.values)
    else:
        y_train = y_train.loc[:, 'cur_label'].values
        y_test = y_test.loc[:, 'cur_label'].values

    # with open('train_test_data' + '.pickle', 'rb') as f:
    #     [train_data, test_data] = pickle.load(f)

    if pca_dim > 0:
        pca_model = PCA(n_components=min(pca_dim, X_train.shape[1])).fit(X_train)
        # plt.plot(pca_model.eig)
        X_train = pca_model.transform(X_train)
        X_test = pca_model.transform(X_test)


    from lightgbm import LGBMRegressor
    clf = LGBMRegressor(boosting_type='gbdt',
		num_leaves=31,
		max_depth=-1,
		learning_rate=0.001,
		n_estimators=1000,
		objective=None,
		min_split_gain=0,
		min_child_weight=3,
		min_child_samples=10,
		subsample=1,
		subsample_freq=1,
		colsample_bytree=0.7,
		reg_alpha=0.3,
		reg_lambda=0,
		seed=17)

    if label_type == 'valence':
        print('VALENCE')
        for i in combs:
            print('train {}, test {}'.format(cleaned(i[0]), cleaned(i[1])))
            [X_temp_train, y_temp_train] = cut_extreme_values(y_train[:, 0], X_train, i[0][0], i[0][1])

            [X_temp_test, y_temp_test] = cut_extreme_values(y_test[:, 0], X_test, i[1][0], i[1][1])

            clf.fit(X_temp_train, y_temp_train)
            y_pred = clf.predict(X_temp_test)
            print('mae= {}'.format(round(mean_absolute_error(y_pred, y_temp_test),3)))
    elif label_type == 'arousal':
        print('AROUSAL')
        for i in combs:
            print('train {}, test {}'.format(cleaned(i[0]), cleaned(i[1])))
            [X_temp_train, y_temp_train] = cut_extreme_values(y_train[:, 1], X_train, i[0][0], i[0][1])

            [X_temp_test, y_temp_test] = cut_extreme_values(y_test[:, 1], X_test, i[1][0], i[1][1])

            clf.fit(X_temp_train, y_temp_train)
            y_pred = clf.predict(X_temp_test)
            print('mae= {}'.format(round(mean_absolute_error(y_pred, y_temp_test), 3)))


def classification(X_train, X_test, y_train, y_test, label_type, pca_dim=100):
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train, X_test]))

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if label_type!='labels':
        scaler = MinMaxScaler()
        temp1, temp2 = y_train.loc[:,['valence','activation']], y_test.loc[:,['valence','activation']]
        temp = pd.concat((temp1, temp2))
        scaler.fit(temp.values)

        y_train = scaler.transform(temp1.values)
        y_test = scaler.transform(temp2.values)
    else:
        y_train = y_train.loc[:,'cur_label'].values
        y_test = y_test.loc[:,'cur_label'].values


    if pca_dim > 0:
        pca_model = PCA(n_components=min(pca_dim, X_train.shape[1])).fit(np.array(X_train))
        X_train = pca_model.transform(np.array(X_train))
        X_test = pca_model.transform(np.array(X_test))

    with open('train_test_data' + '.pickle', 'rb') as f:
        [train_data, test_data] = pickle.load(f)

    # with open('best_rf_cl' + '.pickle', 'rb') as f:
    #     clf = pickle.load(f)
    clf = LGBMClassifier(boosting_type='gbdt',
		num_leaves=31,
		max_depth=-1,
		learning_rate=0.001,
		n_estimators=1000,
		objective=None,
		min_split_gain=0,
		min_child_weight=3,
		min_child_samples=10,
		subsample=0.8,
		subsample_freq=1,
		colsample_bytree=0.7,
		reg_alpha=0.3,
		reg_lambda=0,
		seed=17)


    if label_type=='valence':
        print('VALENCE')
        for i in combs:
            print('train {}, test {}'.format(cleaned(i[0]),cleaned(i[1])))
            [X_temp_train, y_temp_train] = cut_extreme_values(y_train[:, 0], X_train, i[0][0], i[0][1])
            y_temp_train = [extreme_features(x) for x in y_temp_train]


            [X_temp_test, y_temp_test] = cut_extreme_values(y_test[:, 0],X_test, i[1][0], i[1][1])
            y_temp_test = [extreme_features(x) for x in y_temp_test]


            clf.fit(X_temp_train, y_temp_train)
            y_pred = clf.predict(X_temp_test)
            print('f1_score= {}'.format(round(f1_score(y_pred, y_temp_test,average='macro'),3)))
            # print(classification_report(y_pred, y_test))

        # y_test = [extreme_features(x) for x in y_test]
    elif label_type=='arousal':
        print('AROUSAL')
        for i in combs:
            print('train {}, test {}'.format(cleaned(i[0]),cleaned(i[1])))
            [X_temp_train, y_temp_train] = cut_extreme_values(y_train[:, 1], X_train, i[0][0], i[0][1])
            y_temp_train = [extreme_features(x) for x in y_temp_train]

            [X_temp_test, y_temp_test] = cut_extreme_values(y_test[:, 1], X_test, i[1][0], i[1][1])
            y_temp_test = [extreme_features(x) for x in y_temp_test]

            clf.fit(X_temp_train, y_temp_train)
            y_pred = clf.predict(X_temp_test)
            print('f1_score= {}'.format(round(f1_score(y_pred, y_temp_test,average='macro'),3)))
            # print(classification_report(y_pred, y_test))

    elif label_type=='labels':

        # STRATIFICATION
        # X = np.hstack((X_train, y_train[:, np.newaxis]))
        # X_pd = pd.DataFrame(X)
        # X_new = np.zeros((1, X.shape[1]))
        # max_num = max(X_pd.iloc[:, -1].value_counts())
        # for label in np.unique(y_train):
        #     indexes_to_add = np.random.choice(a=X_pd[X_pd.iloc[:, -1] == label].index, size=(max_num,))
        #     X_additional = X_pd.loc[indexes_to_add, :]
        #     X_new = np.vstack((X_new, X_additional.values))
        # X_new = X_new[1:,:]
        # X_train = X_new[:,:-1]
        # y_train = X_new[:, -1]
        # # ============================================================
        # shuffle

        combined = list(zip(X_train, y_train))
        random.shuffle(combined)
        X_train[:], y_train[:] = zip(*combined)


        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print('f1_score= {}'.format(f1_score(y_pred, y_test, average='macro')))
    else:
        raise Exception('label_type is mistaken')


def plot_decision_boundaries(train_targets_omg, train_targets_iemocap):
    matplotlib.use("TkAgg")

    scaler = MinMaxScaler()

    train_targets_omg_omg = train_targets_omg[train_targets_omg['cur_label']!='disgust'][train_targets_omg['cur_label']!='fear'][train_targets_omg['cur_label']!='surprise']

    X = scaler.fit_transform(train_targets_omg.loc[:, ['valence', 'arousal']].values) * 2 - 1

    y = train_targets_omg.loc[:, 'cur_label'].values[:, np.newaxis]

    dict_emo = {}
    reverse_dict_emo = {}
    for idx, i in enumerate(np.unique(train_targets_omg.loc[:, 'cur_label'])):
        dict_emo[i] = idx
        reverse_dict_emo[idx] = i

    labels = np.array([dict_emo[x] for x in train_targets_omg.loc[:, 'cur_label']])[:, np.newaxis]

    y=labels

    temp = pd.DataFrame(data=np.concatenate((X, y), axis=1), columns=['valence', 'arousal', 'cur_label'])

    # Training classifiers
    # clf1 = DecisionTreeClassifier(max_depth=5)
    clf2 = SVC(gamma=.1, kernel='rbf', probability=True)
    clf3 = SVC(gamma=.1, kernel='rbf', probability=True)
    # eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
    #                                     ('svc', clf3)],
    #                         voting='soft', weights=[2, 1, 3])

    # clf1.fit(X, y)
    clf2.fit(X, y)
    clf3.fit(X, y)
    # eclf.fit(X, y)

    # Plotting decision regions
    xx, yy = np.meshgrid(np.arange(-1, 1.1, 0.1),
                         np.arange(-1, 1.1, 0.1))
    print(dict_emo)
    colors = ['brown', 'gold', 'lightgray', 'navy']
    legend_dict = { 'sad': 'navy', 'neutral': 'lightgray', 'happy': 'gold',
                   'anger': 'brown'}


    f, axarr = plt.subplots(1, 2, figsize=(15, 7))

    for idx, clf, tt in zip((0,1),
                            [clf2, clf3],

                            ['KNN (k=5)',
                             'Kernel SVM']):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx].contourf(xx, yy, Z,
                           cmap = matplotlib.colors.ListedColormap(colors), alpha=0.9)
        axarr[idx].scatter(X[:, 0][:,np.newaxis], X[:, 1][:,np.newaxis], c=y,
                                      s=2, edgecolor='k',alpha=0.1)
        axarr[idx].set_title(tt)
        # ======================================================
        for label in temp.cur_label.unique():
            x_t = temp[temp['cur_label'] == label].valence.mean()
            y_t = temp[temp['cur_label'] == label].arousal.mean()
            axarr[idx].plot(x_t,
                                       y_t, color='red', marker='o',
                                       markersize=5)
            axarr[idx].annotate('%s' % reverse_dict_emo[label], xy=(x_t,y_t), textcoords='data')

        patchList = []
        for key in legend_dict:
            data_key = mpatches.Patch(color=legend_dict[key], label=key)
            patchList.append(data_key)
        axarr[idx].legend(handles=patchList, loc=1, fontsize=10)
        axarr[idx].add_artist(plt.Circle((0, 0), 1, color='black', fill=False))
        axis_font = {'fontname': 'Arial', 'size': '12'}
        axarr[idx].set_xlabel('Valence', **axis_font)
        axarr[idx].set_ylabel('Arousal', **axis_font)
        axarr[idx].set_ylim(-1, 1)
        axarr[idx].set_xlim(-1, 1)

    plt.show()
    # plt.savefig('omg_iemocap_general_desigion_boundaries.png')

def plot_valence_arousal(train_targets_omg):
    dict_emo = {}
    reverse_dict_emo = {}
    for idx,i in enumerate(np.unique(train_targets_omg.loc[:, 'cur_label'])):
        dict_emo[i] = idx
        reverse_dict_emo[idx] = i

    labels = np.array([dict_emo[x] for x in train_targets_omg.loc[:, 'cur_label']])[:, np.newaxis]

    scaler = MinMaxScaler()

    valence = scaler.fit_transform(train_targets_omg.loc[:, 'valence'].values.reshape(-1,1))*2-1
    arousal = scaler.fit_transform(train_targets_omg.loc[:, 'arousal'].values.reshape(-1,1))*2-1

    names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(dict_emo)

    colors = ['red', 'fuchsia', 'lime', 'gold', 'lightgray', 'navy', 'aqua']
    legend_dict = {'surprise': 'aqua', 'sad': 'navy', 'neutral': 'lightgray', 'happy': 'gold',
                   'fear': 'lime', 'disgust': 'fuchsia', 'anger': 'red'}


    # colors = ['red', 'purple', 'darkgreen', 'gold', 'lightgray', 'navy', 'aqua', 'pink', 'blue', 'black']
    # legend_dict = {'surprise': 'aqua', 'sad': 'navy', 'neutral': 'lightgray', 'happy': 'gold',
    #                'fear': 'darkgreen', 'disgust': 'purple', 'anger': 'red'}



    matplotlib.use("TkAgg")
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.scatter(valence, arousal, c=labels, cmap=matplotlib.colors.ListedColormap(colors), s=15)
    # cmap = matplotlib.colors.ListedColormap(colors)


    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)
    plt.legend(handles=patchList, loc=1, fontsize=15)
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False))

    matplotlib.rcParams.update({'font.size': 22})

    axis_font = {'fontname': 'Arial', 'size': '20'}
    plt.xlabel('Valence', **axis_font)
    plt.ylabel('Arousal', **axis_font)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    ax.set_title("Emotions dependency on valence/arousal", fontsize=15)
    plt.grid()
    plt.show()
    # plt.savefig('foo3.png')

def plot_decision_boundaries_binary_clf(train_targets):
    matplotlib.use("TkAgg")

    scaler = MinMaxScaler()
    scaler.fit(train_targets[:, 1:])
    X = scaler.transform(train_targets[:, 1:]) * 2 - 1
    y = train_targets[:, 0][:,np.newaxis]

    temp = pd.DataFrame(data=np.concatenate((X,y), axis=1), columns=['valence','arousal','target'])

    # print(temp[temp['target']==1].shape)



    xx, yy = np.meshgrid(np.arange(-1, 1, 0.1),
                         np.arange(-1, 1, 0.1))

    f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(10, 15))
    names = ['Anger','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    for idx, i, name in zip(product([0,1,2,3], [0, 1]),
                            [0,1,2,3,4,5,6], names):

        temp_curr = temp.copy()
        temp_curr['target'] = [1 if x==i else 7 for x in temp_curr.target]
        temp_curr = temp_curr.values

        clf = SVC(gamma=.1, kernel='rbf', probability=True)
        clf.fit(temp_curr[:, :2], temp_curr[:, 2])

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.9)
        axarr[idx[0], idx[1]].scatter(temp_curr[:, 0], temp_curr[:, 1], c=temp_curr[:, 2],
                                      s=2, edgecolor='k',alpha=0.1)
        axarr[idx[0], idx[1]].set_title(name)

    # plt.show()
        plt.savefig('foo1.png')

def plot_decision_boundaries_multiclass_clf(train_targets, test_targets):

    targets = pd.concat([train_targets, test_targets])
    matplotlib.use("TkAgg")

    scaler = MinMaxScaler()

    X = scaler.fit_transform(targets.loc[:, ['valence', 'arousal']].values) * 2 - 1
    y = targets.loc[:, 'cur_label'].values[:, np.newaxis]

    temp = pd.DataFrame(data=np.concatenate((X, y), axis=1), columns=['valence', 'arousal', 'cur_label'])

    clf = SVC(C=1,gamma=0.1, kernel='rbf', probability=True)
    clf.fit(X, y)

    xx, yy = np.meshgrid(np.arange(-1, 1, 0.1),
                         np.arange(-1, 1, 0.1))
    names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(8, 15))


    for idx, label, name in zip(product([0,1,2,3], [0,1]),
                                ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], names):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = np.array([1 if x==label else 0 for x in Z])
        Z = Z.reshape(xx.shape)
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.9)


        axarr[idx[0], idx[1]].scatter(temp[temp['cur_label'] == label].valence,
                                      temp[temp['cur_label'] == label].arousal,

                                      s=2, edgecolor='k', alpha=0.5)
        axarr[idx[0], idx[1]].set_title(name)

        x_test_dot = temp[temp['cur_label'] == label].iloc[:test_targets.shape[0], :].valence.mean()
        y_test_dot = temp[temp['cur_label'] == label].iloc[:test_targets.shape[0], :].arousal.mean()
        axarr[idx[0], idx[1]].plot(x_test_dot, y_test_dot, color='lime', marker='o', markersize=5)
        axarr[idx[0], idx[1]].annotate('%s' % 'test', xy=(x_test_dot, y_test_dot),
                                       textcoords='offset pixels', xytext=(x_test_dot - 30, y_test_dot - 10))

        x_train_dot = temp[temp['cur_label'] == label].iloc[:train_targets.shape[0],:].valence.mean()
        y_train_dot = temp[temp['cur_label'] == label].iloc[:train_targets.shape[0],:].arousal.mean()
        axarr[idx[0], idx[1]].plot(x_train_dot, y_train_dot, color='red', marker='o',markersize=5)
        axarr[idx[0], idx[1]].annotate('%s' % 'train', xy=(x_train_dot, y_train_dot),
                                       textcoords='offset pixels', xytext=(x_train_dot+5, y_train_dot))

    plt.show()
    # plt.savefig('omg_multiclass_all.png')

def get_data(dataset_root, file_list, max_num_clips=0):
    dataset_parser = AVDBParser(dataset_root, file_list)


    data = dataset_parser.get_data()

    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data


if __name__ == "__main__":

    combs = list(combinations_with_replacement([(0.5,0.5),(0.3,0.7)], 2)) + [((0.3,0.7),(0.5,0.5))]

    def cut_extreme_values(y_train, X_train, lower=0.3, upper=0.7):
        right_indexs = []
        temp = pd.Series(y_train)
        for idx, i in enumerate(y_train):
            if i < lower or i > upper:
                right_indexs.append(idx)
        print('данных до обрезания {}, после {}'.format(X_train.shape[0], len(right_indexs)))
        X_train = X_train[right_indexs, :]
        y_train = temp.loc[right_indexs].values
        return [X_train, y_train]

    def cleaned(x):
        if x==(0.5,0.5):
            return 'original'
        else:
            return 'cleaned'

    def extreme_features(x, higher=0.5, lower=0.5):
        if x>=higher:
            return 1
        elif x<=lower:
            return 0

    # CHANGE DIR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    base_dir = r'C:\Users\kotov-d\Documents\базы\iemocap'
    use_features = True
    features_path = os.path.join(base_dir,'feature','opensmile')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    dataset_root = base_dir + '/data'
    # opensmile configuration
    opensmile_root_dir = 'C:/Users/kotov-d/Documents/task#1/opensmile-2.3.0'
    opensmile_config_path = 'C:/Users/kotov-d/Documents/task#1/opensmile-2.3.0/config/avec2013.conf'

    #Data preprocessing
    # if not os.path.exists(os.path.join(base_dir,'train_test_data' + '.pickle')):
    #     with open(os.path.join(features_path, 'f_train.pkl'), 'rb') as f:
    #         train_file_list = pickle.load(f)
    #     with open(os.path.join(features_path, 'f_test.pkl'), 'rb') as f:
    #         test_file_list = pickle.load(f)
    #
    #     train_data = get_data(dataset_root, train_file_list)
    #     test_data = get_data(dataset_root, test_file_list)
    #
    #     with open(os.path.join(base_dir,'train_test_data' + '.pickle'), 'wb') as f:
    #         pickle.dump([train_data, test_data], f, protocol=2)
    # else:
    #     with open(os.path.join(base_dir, 'train_test_data' + '.pickle'), 'rb') as f:
    #         [train_data, test_data] = pickle.load(f)

    # Define x_train, x_test, y_train, y_test
    if use_features==True:
        with open(os.path.join(features_path, 'x_train.pkl'), 'rb') as f:
            x_train = pickle.load(f)
        with open(os.path.join(features_path, 'x_test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
        with open(os.path.join(features_path, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(features_path, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)

    # else:
    #     x_train = train_data
    #     x_test = test_data
    #     y_train = pd.read_csv(train_file_list).loc[:, ['valence', 'activation', 'dominance', 'cur_label']]
    #     y_test = pd.read_csv(test_file_list).loc[:, ['valence', 'activation', 'dominance', 'cur_label']]

    # MAIN PART

    # classification(x_train, x_test, y_train, y_test, label_type='labels', pca_dim=0)

    # classification(x_train, x_test, y_train, y_test, label_type='valence', pca_dim=0)
    # classification(x_train, x_test, y_train, y_test, label_type='arousal', pca_dim=0)

    # regression(x_train, x_test, y_train, y_test, label_type='valence', pca_dim=0)
    # regression(x_train, x_test, y_train, y_test, label_type='arousal', pca_dim=0)


    # ===============================================================
    # plot_valence_arousal(y_train)
    plot_decision_boundaries(y_train, y_test)

    # plot_decision_boundaries_binary_clf(y_train)

    # plot_decision_boundaries_multiclass_clf(y_train, y_test)
