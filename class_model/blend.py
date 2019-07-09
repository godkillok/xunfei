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
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import os
import sys
import logging
currentUrl = os.path.dirname(__file__)
most_parenturl = os.path.abspath(os.path.join(currentUrl, os.pardir))
m_p, m_c = os.path.split(most_parenturl)
while 'xunfei' not in m_c:
    m_p, m_c = os.path.split(m_p)
sys.path.append(os.path.join(m_p, m_c))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from class_model.load_data import  load_data
from sklearn.calibration import CalibratedClassifierCV

def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) +
                     (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set1

    n_folds = 10
    shuffle = False

    train_x, train_y, test_x, test_y, pred_x, apps, label_dic= load_data()
    if "no" in label_dic:
        del label_dic["no"]
    logging.info("======"*20)
    logging.info((len(train_x),len(test_x)))
    if shuffle:
        idx = np.random.permutation(train_y.size)
        train_x = train_x[idx]
        train_y = train_y[idx]

    sfolder = StratifiedKFold( n_folds)
    skf =list(sfolder.split(train_x, train_y))
    lin_clf = CalibratedClassifierCV(svm.LinearSVC(C=0.00001))
    lin_clf = CalibratedClassifierCV(svm.LinearSVC(C=0.1))
    lin_clf_1 = CalibratedClassifierCV(svm.LinearSVC(C=0.2))
    lin_clf_2 = CalibratedClassifierCV(svm.LinearSVC(C=0.5))
    lin_clf_3 = CalibratedClassifierCV(svm.LinearSVC(C=1))
    lin_clf_4 = CalibratedClassifierCV(svm.LinearSVC(C=10))
    tfidf_vec2 = TfidfVectorizer(ngram_range=(1,2), min_df=10, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    tfidf_vec1 = TfidfVectorizer(ngram_range=(1,1), min_df=10, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)

    tfidf_vec3 = TfidfVectorizer(ngram_range=(1,3), min_df=10, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    tfidf_vec4 = TfidfVectorizer(ngram_range=(1, 4), min_df=10, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    data_set=train_x+test_x+pred_x
    tfidf_vec1.fit_transform(data_set)
    tfidf_vec4.fit_transform(data_set)
    tfidf_vec3.fit_transform(data_set)
    tfidf_vec2.fit_transform(data_set)
    # with open(project_path + 'tfidf_vec1.pkl', 'wb') as f:
    #     pickle.dump(tfidf_vec1, f)
    # with open(project_path + 'tfidf_vec2.pkl', 'wb') as f:
    #     pickle.dump(tfidf_vec2, f)
    # with open(project_path + 'tfidf_vec3.pkl', 'wb') as f:
    #     pickle.dump(tfidf_vec3, f)
    # with open(project_path + 'tfidf_vec4.pkl', 'wb') as f:
    #     pickle.dump(tfidf_vec4, f)
    with open(project_path + 'tfidf_vec1.pkl', 'rb') as f:
        tfidf_vec1 = pickle.load(f)
    with open(project_path + 'tfidf_vec2.pkl', 'rb') as f:
        tfidf_vec2 = pickle.load(f)
    with open(project_path + 'tfidf_vec3.pkl', 'rb') as f:
        tfidf_vec3 = pickle.load(f)
    with open(project_path + 'tfidf_vec4.pkl', 'rb') as f:
        tfidf_vec4 = pickle.load(f)
    clfs=[]
    for i  in [1e-5,1e-4,1e-1,1]:
        clfs.append([CalibratedClassifierCV(svm.LinearSVC(C=c)),tfidf_vec3])



    # lin_clf.fit(trn_term_doc, train_y)
    logging.info ("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((len(train_x), len(clfs),len(label_dic)))
    dataset_blend_pred = np.zeros((len(pred_x), len(clfs),len(label_dic)))
    dataset_blend_test = np.zeros((len(test_x), len(clfs),len(label_dic)))
    logging.info("dataset_blend_test {}".format(dataset_blend_test.shape))
    for j, clf_process in enumerate(clfs):
        clf,process=clf_process

        logging.info((j, clf))
        dataset_blend_pred_j = np.zeros((len(pred_x),len(skf),len(label_dic)))
        dataset_blend_test_j = np.zeros((len(test_x), len(skf),len(label_dic)))
        for i, (train_1, test_1) in enumerate(skf):
            logging.info (("Fold", i))
            logging.info(train_1)
            X_train = np.array(train_x)[train_1]
            y_train = np.array(train_y)[train_1]
            X_test = np.array(train_x)[test_1]
            y_test =  np.array(train_y)[test_1]
            train_x_doc = process.transform(X_train)
            test_x_doc = process.transform(X_test)
            pred_x_doc=process.transform(pred_x)
            test_xx_doc=process.transform(test_x)
            print(y_train)
            clf.fit(train_x_doc, y_train)
            y_test_prob = clf.predict_proba(test_x_doc)[:,:]
            logging.info("y_test_prob {}".format(y_test_prob.shape))
            dataset_blend_train[test_1,j, :] = y_test_prob
            dataset_blend_pred_j[:, i,:] = clf.predict_proba(pred_x_doc)[:, :]
            dataset_blend_test_j[:, i,:] = clf.predict_proba(test_xx_doc)[:, :]
        logging.info("dataset_blend_pred_j mean (sample,prob) {} ".format(np.mean(dataset_blend_pred_j,axis=1).shape))
        dataset_blend_pred[:, j,:] =np.mean(dataset_blend_pred_j,axis=1)
        dataset_blend_test[:, j,:] =np.mean(dataset_blend_test_j,axis=1)

    logging.info((len(pred_x),len(skf),len(label_dic)))
    logging.info('dataset_blend_train1 {}'.format(dataset_blend_train.shape)) #size of train sample, size of clf, size of class
    logging.info('dataset_blend_pred {}'.format(dataset_blend_pred.shape))
    dataset_blend_train=dataset_blend_train.reshape(dataset_blend_train.shape[0],dataset_blend_train.shape[1]*dataset_blend_train.shape[2])
    dataset_blend_pred = dataset_blend_pred.reshape(dataset_blend_pred.shape[0],
                                                    dataset_blend_pred.shape[1] * dataset_blend_pred.shape[2])
    dataset_blend_test = dataset_blend_test.reshape(dataset_blend_test.shape[0],
                                                    dataset_blend_test.shape[1] * dataset_blend_test.shape[2])
    logging.info('dataset_blend_train2 {}'.format(dataset_blend_train.shape)) #size of train sample, size of clf*size of class

    logging.info ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, train_y)

    #clf.predict_proba(dataset_blend_test)[:, 1]
    gg=clf.predict_proba(dataset_blend_pred)
    dataset_blend_test_prob=clf.predict_proba(dataset_blend_test)
    test_preds_=clf.predict(dataset_blend_test)
    logging.info('accuracy_score {} top1 test\n {}'.format(accuracy_score(test_y, test_preds_),
                                                    classification_report(test_y,
                                                                          test_preds_)))
    test_preds=[]
    for prob in dataset_blend_test_prob:
        test_preds.append(list(prob.argsort()[-2:][::-1]))
    test_preds_=[]
    for rea,tes in zip(test_y,test_preds):
        prd=tes[0]
        for te in tes:
            if rea==te:
                prd=te
        test_preds_.append(prd)
    logging.info('accuracy_score {} top2 test\n {}'.format( accuracy_score(test_y,test_preds_),
                                                                               classification_report(test_y,
                                                                                                     test_preds_)))
    logging.info(gg.shape)
    y_submission = gg[:, 1]
    logging.info(y_submission.shape)
    logging.info ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    logging.info ("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission)+1), y_submission]).T
    logging.info(tmp.shape)
    np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
               header='MoleculeId,PredictedProbability', comments='')
