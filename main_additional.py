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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def regression(X_train, X_test, y_train, y_test, label_type, pca_dim=100):
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train, X_test]))

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if label_type != 'labels':
        scaler = MinMaxScaler()
        temp1, temp2 = pd.DataFrame(y_train), pd.DataFrame(y_test)
        temp = pd.concat((temp1, temp2))
        scaler.fit(temp.iloc[:, 1:].values)

        y_train, y_test = np.zeros((temp1.shape[0], temp1.shape[1])), np.zeros((temp2.shape[0], temp2.shape[1]))
        y_train[:, 1:] = scaler.transform(temp1.iloc[:, 1:].values)
        y_test[:, 1:] = scaler.transform(temp2.iloc[:, 1:].values)

    with open('train_test_data' + '.pickle', 'rb') as f:
        [train_data, test_data] = pickle.load(f)

    if pca_dim > 0:
        pca_model = PCA(n_components=min(pca_dim, X_train.shape[1])).fit(X_train)
        # plt.plot(pca_model.eig)
        X_train = pca_model.transform(X_train)
        X_test = pca_model.transform(X_test)

    accuracy_fn = Accuracy_regression(test_data, label_type=label_type, scaler=scaler)

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
            [X_temp_train, y_temp_train] = cut_extreme_values(y_train[:, 1], X_train, i[0][0], i[0][1])

            [X_temp_test, y_temp_test] = cut_extreme_values(y_test[:, 1], X_test, i[1][0], i[1][1])

            clf.fit(X_temp_train, y_temp_train)
            y_pred = clf.predict(X_temp_test)
            print('mae= {}'.format(round(mean_absolute_error(y_pred, y_temp_test),3)))
    elif label_type == 'arousal':
        print('AROUSAL')
        for i in combs:
            print('train {}, test {}'.format(cleaned(i[0]), cleaned(i[1])))
            [X_temp_train, y_temp_train] = cut_extreme_values(y_train[:, 2], X_train, i[0][0], i[0][1])

            [X_temp_test, y_temp_test] = cut_extreme_values(y_test[:, 2], X_test, i[1][0], i[1][1])

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


def plot_decision_boundaries(train_targets):
    matplotlib.use("TkAgg")

    scaler = MinMaxScaler()

    train_targets = train_targets[train_targets['cur_label']!='fea'][train_targets['cur_label']!='oth'][train_targets['cur_label']!='xxx']\
        [train_targets['cur_label']!='sur'][train_targets['cur_label']!='exc'][train_targets['cur_label']!='fru'][train_targets['cur_label']!='dis']

    train_targets.rename(columns={'activation':'arousal'},inplace=True)
    X = scaler.fit_transform(train_targets.loc[:, ['valence', 'arousal']].values) * 2 - 1

    y = train_targets.loc[:, 'cur_label'].values[:, np.newaxis]

    dict_emo = {}
    reverse_dict_emo = {}
    for idx, i in enumerate(np.unique(train_targets.loc[:, 'cur_label'])):
        dict_emo[i] = idx
        reverse_dict_emo[idx] = i

    labels = np.array([dict_emo[x] for x in train_targets.loc[:, 'cur_label']])[:, np.newaxis]

    y=labels

    temp = pd.DataFrame(data=np.concatenate((X, y), axis=1), columns=['valence', 'arousal', 'cur_label'])

    # Training classifiers
    clf1 = DecisionTreeClassifier(max_depth=5)
    clf2 = KNeighborsClassifier(n_neighbors=5)
    clf3 = SVC(gamma=.1, kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                        ('svc', clf3)],
                            voting='soft', weights=[2, 1, 3])

    clf1.fit(X, y)
    clf2.fit(X, y)
    clf3.fit(X, y)
    eclf.fit(X, y)

    # Plotting decision regions
    xx, yy = np.meshgrid(np.arange(-1, 1.1, 0.1),
                         np.arange(-1, 1.1, 0.1))
    print(dict_emo)
    colors = ['brown', 'orange', 'lightgray', 'navy']
    legend_dict = { 'sad': 'navy', 'neutral': 'lightgray', 'anger': 'brown', 'happy': 'orange'}


    plt.figure(figsize=(7, 7))

    Z = clf3.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                       cmap = matplotlib.colors.ListedColormap(colors), alpha=0.9)
    plt.scatter(X[:, 0][:,np.newaxis], X[:, 1][:,np.newaxis], c=y,
                                  s=2, edgecolor='k',alpha=0.1)
    plt.title('IEMOCAP video segmentation')
    # ======================================================
    for label in temp.cur_label.unique():
        x_t = temp[temp['cur_label'] == label].valence.mean()
        y_t = temp[temp['cur_label'] == label].arousal.mean()
        plt.plot(x_t,
                                   y_t, color='red', marker='o',
                                   markersize=5)
        plt.annotate('%s' % reverse_dict_emo[label], xy=(x_t,y_t), textcoords='data')

    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)
    plt.legend(handles=patchList, loc=1, fontsize=10)
    # plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gcf().gca().add_artist(plt.Circle((0, 0), 1, color='black', fill=False))
    axis_font = {'fontname': 'Arial', 'size': '12'}
    plt.xlabel('Valence', **axis_font)
    plt.ylabel('Arousal', **axis_font)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)

    # plt.show()
    plt.savefig('iemocap_general_desigion_boundaries.png')


def plot_valence_arousal(train_targets):
    train_targets = train_targets[train_targets['cur_label'] != 'xxx'][train_targets['cur_label'] != 'dis'][\
        train_targets['cur_label'] != 'fea'][train_targets['cur_label'] != 'oth']

    dict_emo = {}
    reverse_dict_emo = {}
    for idx,i in enumerate(np.unique(train_targets.loc[:, 'cur_label'])):
        dict_emo[i] = idx
        reverse_dict_emo[idx] = i

    labels = np.array([dict_emo[x] for x in train_targets.loc[:, 'cur_label']])[:, np.newaxis]

    scaler = MinMaxScaler()

    valence = scaler.fit_transform(train_targets.loc[:, 'valence'].values.reshape(-1,1))*2-1
    arousal = scaler.fit_transform(train_targets.loc[:, 'activation'].values.reshape(-1,1))*2-1

    names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(dict_emo)

    # colors = ['red', 'fuchsia', 'white', 'lime', 'white', 'gold', 'lightgray', 'navy', 'aqua', 'white']
    # legend_dict = {'surprise': 'aqua', 'sad': 'navy', 'neutral': 'lightgray', 'happy': 'gold',
    #                'fear': 'lime', 'disgust': 'fuchsia', 'anger': 'red'}


    colors = ['red', 'purple', 'darkgreen', 'gold', 'lightgray', 'navy', 'aqua', 'pink', 'blue', 'black']
    legend_dict = {'surprise': 'aqua', 'sad': 'navy', 'neutral': 'lightgray', 'happy': 'gold',
                   'fear': 'darkgreen', 'disgust': 'purple', 'anger': 'red'}



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
    plt.ylabel('Activation', **axis_font)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    ax.set_title("Emotions dependency on valence/activation", fontsize=15)
    plt.grid()
    plt.show()
    # plt.savefig('iemocap_val_arousal_plot.png')

def plot_decision_boundaries_binary_clf(train_targets):
    matplotlib.use("TkAgg")

    scaler = MinMaxScaler()
    print(train_targets.columns)

    X = scaler.fit_transform(train_targets.loc[:, ['valence','activation']].values)*2-1
    y = train_targets.loc[:, 'cur_label'].values[:,np.newaxis]

    temp = pd.DataFrame(data=np.concatenate((X,y), axis=1), columns=['valence','activation','cur_label'])


    xx, yy = np.meshgrid(np.arange(-1, 1, 0.1),
                         np.arange(-1, 1, 0.1))

    f, axarr = plt.subplots(5, 2, sharex='col', sharey='row', figsize=(10, 15))
    names = ['Anger','Disgust','Exc','Fear','Fru','Happy','Neutral','Sad','Surprise','xxx']
    for idx, i, name in zip(product([0,1,2,3,4], [0, 1]),
                            ['ang','dis','exc','fea','fru','hap','neu','sad','sur','xxx'], names):


            temp_curr = temp.copy()
            temp_curr.loc[:,'cur_label'] = [1 if x==i else 0 for x in temp_curr.cur_label]

            clf = SVC(gamma=.1, kernel='rbf', probability=True)
            clf.fit(temp_curr.loc[:, ['valence','activation']], temp_curr.loc[:, 'cur_label'])

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.9)
            axarr[idx[0], idx[1]].scatter(temp_curr.loc[:,'valence'], temp_curr.loc[:,'activation'], c=temp_curr.loc[:,'cur_label'],
                                          s=2, edgecolor='k',alpha=0.1)
            axarr[idx[0], idx[1]].set_title(name)



    # plt.show()
    plt.savefig('iemocap_svm_binary.png')

def plot_3d_version(train_targets,test_targets):
    train_targets = train_targets[train_targets['cur_label'] != 'xxx'][train_targets['cur_label'] != 'oth'][
        train_targets['cur_label'] != 'dis']
    test_targets = test_targets[test_targets['cur_label'] != 'xxx'][test_targets['cur_label'] != 'oth'][
        test_targets['cur_label'] != 'dis']

    targets = pd.concat([train_targets, test_targets])

    dict_emo, reverse_dict_emo = {}, {}
    for idx, i in enumerate(np.unique(targets.loc[:, 'cur_label'])):
        dict_emo[i] = idx
        reverse_dict_emo[idx] = i

    labels = np.array([dict_emo[x] for x in targets.loc[:, 'cur_label']])[:, np.newaxis]

    # preprocessing ================================================================================
    scaler = MinMaxScaler()

    X = scaler.fit_transform(targets.loc[:, ['valence', 'activation', 'dominance']].values) * 2 - 1
    y = targets.loc[:, 'cur_label'].values[:, np.newaxis]

    temp = pd.DataFrame(data=np.concatenate((X, y), axis=1), columns=['valence', 'activation', 'dominance', 'cur_label'])

    clf = SVC(gamma=.01, kernel='rbf', probability=True, decision_function_shape='ovr')
    clf.fit(X, y)

    # plot 3D subplots=============================================================================================
    matplotlib.use("TkAgg")

    fig = plt.figure(figsize=(16, 8))
    names = ['Anger', 'Exc', 'Fear', 'Fru', 'Happy', 'Neutral', 'Sad', 'Surprise']
    for idx, i, name in zip(range(8),  ['ang', 'exc', 'fea', 'fru', 'hap', 'neu', 'sad', 'sur'], names):
        ax = fig.add_subplot(2,4,idx+1, projection='3d')

        # Generate the values
        x_vals = list(temp[temp.cur_label==i].valence)
        y_vals = list(temp[temp.cur_label==i].activation)
        z_vals = list(temp[temp.cur_label==i].dominance)

        # Plot the values
        ax.scatter(x_vals, y_vals, z_vals, c='g', marker='o', s=5)

        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominance')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(name)

    plt.show()
    # plt.savefig('3d_iemocap.jpg')

def plot_decision_boundaries_multiclass_clf(train_targets,test_targets):
    train_targets = train_targets[train_targets['cur_label']!='xxx'][train_targets['cur_label']!='oth'][train_targets['cur_label']!='dis']
    test_targets = test_targets[test_targets['cur_label']!='xxx'][test_targets['cur_label']!='oth'][test_targets['cur_label']!='dis']


    targets = pd.concat([train_targets, test_targets])
    dict_emo = {}
    reverse_dict_emo = {}
    for idx, i in enumerate(np.unique(targets.loc[:, 'cur_label'])):
        dict_emo[i] = idx
        reverse_dict_emo[idx] = i

    labels = np.array([dict_emo[x] for x in targets.loc[:, 'cur_label']])[:, np.newaxis]

    matplotlib.use("TkAgg")

    scaler = MinMaxScaler()


    X = scaler.fit_transform(targets.loc[:, ['valence', 'activation']].values) * 2 - 1
    y = targets.loc[:, 'cur_label'].values[:, np.newaxis]

    temp = pd.DataFrame(data=np.concatenate((X, y), axis=1), columns=['valence', 'activation', 'cur_label'])

    clf = SVC(gamma=.01, kernel='rbf', probability=True, decision_function_shape='ovr')
    clf.fit(X, y)

    xx, yy = np.meshgrid(np.arange(-1, 1, 0.1),
                         np.arange(-1, 1, 0.1))

    f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(8, 15))
    names = ['Anger', 'Exc', 'Fear', 'Fru', 'Happy', 'Neutral', 'Sad', 'Surprise']
    for idx, label, name in zip(product([0, 1, 2, 3], [0, 1]),
                            ['ang', 'exc', 'fea', 'fru', 'hap', 'neu', 'sad', 'sur'], names):


        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array([1 if x == label else 0 for x in Z])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.9)

        axarr[idx[0], idx[1]].scatter(temp[temp['cur_label'] == label].valence,
                                      temp[temp['cur_label'] == label].activation,

                                      s=2, edgecolor='k', alpha=0.5)
        axarr[idx[0], idx[1]].set_title(name)

        x_test_dot = temp[temp['cur_label'] == label].iloc[:test_targets.shape[0], :].valence.mean()
        y_test_dot = temp[temp['cur_label'] == label].iloc[:test_targets.shape[0], :].activation.mean()
        axarr[idx[0], idx[1]].plot(x_test_dot, y_test_dot, color='lime', marker='o', markersize=5)
        axarr[idx[0], idx[1]].annotate('%s' % 'test', xy=(x_test_dot, y_test_dot),
                                       textcoords='offset pixels', xytext=(x_test_dot - 30, y_test_dot - 10))

        x_train_dot = temp[temp['cur_label'] == label].iloc[:train_targets.shape[0], :].valence.mean()
        y_train_dot = temp[temp['cur_label'] == label].iloc[:train_targets.shape[0], :].activation.mean()
        axarr[idx[0], idx[1]].plot(x_train_dot, y_train_dot, color='red', marker='o', markersize=5)
        axarr[idx[0], idx[1]].annotate('%s' % 'train', xy=(x_train_dot, y_train_dot),
                                       textcoords='offset pixels', xytext=(x_train_dot + 5, y_train_dot))
    plt.show()
    # plt.savefig('iemocap_multiclass_aa.png')
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
        # print('{} данных было'.format(X_train.shape[0]))
        X_train = X_train[right_indexs, :]
        # print('{} данных стало'.format(len(right_indexs)))
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

    # plot_decision_boundaries_binary_clf(y_train)
    # plot_decision_boundaries_multiclass_clf(y_train, y_test)
    # plot_3d_version(y_train, y_test)
    plot_decision_boundaries(y_train)
