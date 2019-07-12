
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,HashingVectorizer
from sklearn import svm
import logging
import numpy as np
import time
import os
import pickle  # pickle模块2
import logging
import os
import sys


# Hyperparameters tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
currentUrl = os.path.dirname(__file__)
most_parenturl = os.path.abspath(os.path.join(currentUrl, os.pardir))
m_p, m_c = os.path.split(most_parenturl)
while 'xunfei' not in m_c:
    m_p, m_c = os.path.split(m_p)

sys.path.append(os.path.join(m_p, m_c))
from  class_model.load_data import  load_data,top_2_label_code
from sklearn.metrics import accuracy_score
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

project_path="/data/tanggp/xun_class//aichallenge/"
test_path=os.path.join(project_path,"apptype_train.test_jieba_json")
train_path=os.path.join(project_path,"apptype_train.train_jieba_json")
pred_path=os.path.join(project_path,"app_desc.jieba_json")

label_num = 0
t = time.time()
SEED = 314159265
import json

from sklearn.calibration import CalibratedClassifierCV

train_x, train_y, test_x, test_y, pred_x, apps, label_dic = load_data()
logging.info('train {} test{}'.format(len(train_x), len(test_x)))
data_set = train_x + test_x + pred_x
def score(params):



    t1=time.time()
    logging.info(params)
    vec = TfidfVectorizer(ngram_range=(1,int(params["ngram_range"])), min_df=params["min_df"], max_df=params["max_df"], use_idf=1, smooth_idf=1, sublinear_tf=1)
    #vec=HashingVectorizer(ngram_range=(1, 3))

    vec.fit_transform(data_set)
    #
    # with open(project_path + 'tfidf.pkl', 'wb') as f:
    #     pickle.dump(vec, f)
    # # with open(CHANNEL_MODEL + 'tfidf.pkl', 'rb') as f:
    # #     vec = pickle.load(f)


    trn_term_doc = vec.transform(train_x)

    lin_clf=svm.LinearSVC(C=params["C"], class_weight=None, dual=False,fit_intercept=True, intercept_scaling=params["intercept_scaling"],loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2',random_state=3, verbose=False)
    lin_clf = CalibratedClassifierCV(lin_clf)
    lin_clf.fit(trn_term_doc, train_y)
    train_preds=lin_clf.predict(trn_term_doc)


    acc=accuracy_score(train_y, train_preds)

    logging.info("===" * 8)

    test_term_doc = vec.transform(test_x)
    test_preds = lin_clf.predict(test_term_doc)
    test_preds_prob = lin_clf.predict_proba(test_term_doc)
    acc2 = accuracy_score(test_y, test_preds)
    logging.info((len(test_preds_prob),len(test_y)))

    test_preds=top_2_label_code(test_preds_prob, test_y)


    acc3 = accuracy_score(test_y, test_preds)
    loss = 1 - (acc+acc3)/2
    t2 = time.time()
    logging.info("acc {}, on test  set is {} and top2 acc {},loss {} time {},params: \n{}".format(acc,acc2,acc3,loss,t2-t1,params))
    return {'loss': loss, 'status': STATUS_OK}

def optimize(
             #trials,
             random_state=SEED):
    space = { "ngram_range":hp.quniform("ngram_range",1, 4, 1),
              "C":hp.loguniform("C", np.log(1e-5), np.log(1e5)),
              "min_df": hp.choice('min_df', np.arange(10, 50, dtype=int)),
              "max_df":hp.choice('max_df', np.array([0.9,0.92,0.95])),
              "intercept_scaling":hp.loguniform("intercept_scaling", np.log(1e-1), np.log(1e1))
              }
    best = fmin(score, space, algo=tpe.suggest,
                max_evals=250)
    return best

if __name__ == "__main__":
    best_hyperparams = optimize(
        # trials
    )
    logging.info("The best hyperparameters are: ", "\n")
    logging.info(best_hyperparams)
    #svm_pred()

