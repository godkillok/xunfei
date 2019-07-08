"""Kaggle competition: Predicting a Biological Response.

Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)

Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xunfei.class_model.load_data import  load_data
from sklearn.calibration import CalibratedClassifierCV
from
def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) +
                     (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    train_x, train_y, test_x, test_y, pred_x, apps, label_dic= load_data()

    if shuffle:
        idx = np.random.permutation(train_y.size)
        train_x = train_x[idx]
        train_y = train_y[idx]

    skf = list(StratifiedKFold(train_y, n_folds))

    clfs =[ {RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),},
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    lin_clf = CalibratedClassifierCV(svm.LinearSVC(C=0.1))
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=10, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    #vec=HashingVectorizer(ngram_range=(1, 3))
    data_set=train_x+test_x+pred_x
    vec.fit_transform(data_set)
    #
    with open(project_path + 'tfidf.pkl', 'wb') as f:
        pickle.dump(vec, f)
    # with open(CHANNEL_MODEL + 'tfidf.pkl', 'rb') as f:
    #     vec = pickle.load(f)


    trn_term_doc = vec.transform(train_x)

    lin_clf.fit(trn_term_doc, train_y)
    print ("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((pred_x.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((pred_x.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print ("Fold", i)
            X_train = train_x[train]
            y_train = train_y[train]
            X_test = train_x[test]
            y_test = train_y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(pred_x)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)


    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, train_y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print ("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission)+1), y_submission]).T
    np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
               header='MoleculeId,PredictedProbability', comments='')
