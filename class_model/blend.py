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
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import os
import sys
currentUrl = os.path.dirname(__file__)
most_parenturl = os.path.abspath(os.path.join(currentUrl, os.pardir))
m_p, m_c = os.path.split(most_parenturl)
while 'xunfei' not in m_c:
    m_p, m_c = os.path.split(m_p)
print(m_p,m_c)
sys.path.append(os.path.join(m_p, m_c))

from class_model.load_data import  load_data
from sklearn.calibration import CalibratedClassifierCV

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

    skf = StratifiedKFold( n_folds)
    lin_clf = CalibratedClassifierCV(svm.LinearSVC(C=0.1))
    lin_clf_1 = CalibratedClassifierCV(svm.LinearSVC(C=0.2))
    lin_clf_2 = CalibratedClassifierCV(svm.LinearSVC(C=0.5))
    lin_clf_3 = CalibratedClassifierCV(svm.LinearSVC(C=1))
    lin_clf_4 = CalibratedClassifierCV(svm.LinearSVC(C=10))
    tfidf_vec = TfidfVectorizer(ngram_range=(1,3), min_df=10, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    data_set=train_x+test_x+pred_x
    tfidf_vec.fit_transform(data_set)

    clfs =[ {lin_clf,tfidf_vec},
            {lin_clf_1,tfidf_vec},
          ]
    # {lin_clf_2, tfidf_vec},
    # {lin_clf_3, tfidf_vec},
    # {lin_clf_4, tfidf_vec}




    # lin_clf.fit(trn_term_doc, train_y)
    print ("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)))
    dataset_blend_pred = np.zeros((pred_x.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test_x.shape[0], len(clfs)))

    for j, clf_process in enumerate(clfs):
        clf,process=clf_process
        print (j, clf)
        dataset_blend_pred_j = np.zeros((pred_x.shape[0], len(skf)))
        dataset_blend_test_j = np.zeros((test_x.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print ("Fold", i)
            X_train = train_x[train]
            y_train = train_y[train]
            X_test = train_x[test]
            y_test = train_y[test]
            train_x_doc = process.transform(X_train)
            test_x_doc = process.transform(X_test)
            pred_x_doc=process.transform(pred_x)
            test_xx_doc=process.transform(test_x)

            clf.fit(train_x_doc, y_train)
            y_test = clf.predict_proba(test_x_doc)[:, 1]
            dataset_blend_train[test, j] = y_test
            dataset_blend_pred_j[:, i] = clf.predict_proba(pred_x_doc)[:, 1]
            dataset_blend_test_j[:, i] = clf.predict_proba(test_xx_doc)[:, 1]
        dataset_blend_pred[:, j] = dataset_blend_pred_j.mean(1)
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)


    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, train_y)

    #clf.predict_proba(dataset_blend_test)[:, 1]

    y_submission = clf.predict_proba(dataset_blend_pred)[:, 1]

    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print ("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission)+1), y_submission]).T
    print(tmp.shape)
    np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
               header='MoleculeId,PredictedProbability', comments='')
