from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import itertools
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
import pandas as pd

import torch
import torch.nn.functional as F


class Accuracy:
    def __init__(self, data, scaler, experiment_name='', label_type='labels'):
        super(Accuracy, self).__init__()
        self.experiment_name = experiment_name
        if label_type=='labels':
            self.target_clips = [clip.labels for clip in data]
        else:
            temp = pd.DataFrame(columns=['valence','arousal'])
            temp['valence'] = [clip.valence for clip in data]
            temp['arousal'] = [clip.arousal for clip in data]
            temp = scaler.transform(temp)

            if label_type=='valence':
                self.target_clips = list(temp[:,0])
                self.target_clips = [1 if x >0.5 else 0 for x in self.target_clips]
            elif label_type=='arousal':
                self.target_clips = list(temp[:,1])
                self.target_clips = [1 if x > 0.5 else 0 for x in self.target_clips]
        self.target_clips = np.asarray(self.target_clips, dtype=np.int32)
        self.target_names = sorted([str(int(l)) for l in Counter(self.target_clips).keys()])

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print(title+'\n', cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def calc_cnf_matrix(self, target, predict):
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(target, predict)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        title = 'Confusion matrix'
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=self.target_names, title=title)
        plt.savefig(self.experiment_name + '_' + title + '.png')

        # Plot normalized confusion matrix
        title = 'Normalized confusion matrix'
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=self.target_names, normalize=True, title=title)
        plt.savefig(self.experiment_name + '_' + title + '.png')

    def by_clips(self, predict):
        predict_clips = np.asarray(predict, dtype=np.int32)
        assert self.target_clips.shape[0] == predict_clips.shape[0], 'Invalid predict!'

        print('F1-SCORE_WEIGHTED',f1_score(self.target_clips,predict_clips,average='weighted'))
        print(classification_report(self.target_clips, predict_clips, target_names=self.target_names))
        self.calc_cnf_matrix(self.target_clips, predict_clips)

class Accuracy_regression:
    def __init__(self, data, scaler, label_type):
        super(Accuracy_regression, self).__init__()
        self.target_clips = [[clip.valence, clip.arousal] for clip in data]
        self.target_clips = np.asarray(self.target_clips, dtype=np.float32)
        self.target_names = ['Valence', 'Arousal']

        temp = pd.DataFrame(columns=['valence', 'arousal'])
        temp['valence'] = [clip.valence for clip in data]
        temp['arousal'] = [clip.arousal for clip in data]
        temp = scaler.transform(temp)

        if label_type == 'valence':
            self.target_clips = list(temp[:, 0])
        elif label_type == 'arousal':
            self.target_clips = list(temp[:, 1])
        self.target_clips = np.asarray(self.target_clips, dtype=np.float32)


    def by_clips(self, predict):
        predict_clips = np.asarray(predict, dtype=np.float32)
        assert self.target_clips.shape[0] == predict_clips.shape[0], 'Invalid predict!'

        result = mean_absolute_error(self.target_clips, predict_clips)
        print('MAE = {}'.format(result))
        print('---------\n')
        return result

    def __call__(self, targets, predict):
        return self.by_clips(targets, predict)
