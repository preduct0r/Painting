# class Accuracy_regression:
#     def __init__(self, data, threshold=0.1):
#         super(Accuracy_regression, self).__init__()
#         self.threshold = threshold
#         self.target_clips = [[clip.valence, clip.arousal] for clip in data]
#         self.target_clips = np.asarray(self.target_clips, dtype=np.float32)
#         self.target_names = ['Valence', 'Arousal']
#
#     def by_clips(self, targets, predict):
#         predict_clips = np.asarray(predict, dtype=np.float32)
#
#         result = []
#         for k, name in enumerate(self.target_names):
#             target = torch.from_numpy(self.target_clips[:,k])
#             pred = torch.from_numpy(predict_clips[:,k])
#             test_acc = torch.nonzero(F.relu(-(target - pred).abs_() + self.threshold)).size(0)
#             test_acc *= 100 / self.target_clips.shape[0]
#             test_err = F.relu((target - pred).abs_() - self.threshold)
#             test_err = test_err[test_err.nonzero()]
#             result.append(test_acc)
#             print(name + ':')
#             print('   accuracy per clips: %0.3f%%' % test_acc)
#             print('   error per clips: mean=%0.3f, std=%0.3f' % (test_err.mean(), test_err.std()))
#         print('---------\n')
#         return result


# # GRID SEARCH CV
# parameters = param_grid = {'C':[0.01,0.1,1,5,10,100,1000],
#                                'gamma':[1,0.1,0.001,0.0001],
#                                'kernel':['linear','rbf','poly']}
#     svc = svm.SVC()
#     gscv = GridSearchCV(svc, parameters, cv=3, error_score=0.0)
#     gscv.fit(X_train, y_train)
#     clf = gscv.best_estimator_
#
#     with open('clf' + '.pickle', 'wb') as f:
#             pickle.dump(gscv.best_estimator_, f, protocol=2)