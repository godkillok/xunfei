from hpsklearn import HyperoptEstimator, any_sparse_classifier, tfidf,liblinear_svc
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets

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
import json
import datetime
from collections import defaultdict

currentUrl = os.path.dirname(__file__)
most_parenturl = os.path.abspath(os.path.join(currentUrl, os.pardir))
m_p, m_c = os.path.split(most_parenturl)
while 'xunfei' not in m_c:
    m_p, m_c = os.path.split(m_p)
print(m_p,m_c)
sys.path.append(os.path.join(m_p, m_c))
from  class_model.load_data import  load_data
from sklearn.metrics import accuracy_score
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# 1281312322
column = "word_seg"
project_path="/data/tanggp/xfyun/classify/aichallenge/"
test_path=os.path.join(project_path,"apptype_train.test_jieba_json")
train_path=os.path.join(project_path,"apptype_train.train_jieba_json")
pred_path=os.path.join(project_path,"app_desc.jieba_json")
label_dic = {}
label_num = 0
t = time.time()

import json

from sklearn.calibration import CalibratedClassifierCV

def get_data_set(flie):
    global label_num
    with open(flie) as f:
        lines = f.readlines()
    data_x = []
    data_y = []
    apps=[]
    for li in lines:
        li=json.loads(li)
        text=li.get("jieba")
        label1=li.get("label","no") #label_1st
        app=li.get("app")
        apps.append(app)
        if label1 not in label_dic.keys():
            label_dic[label1] = label_num
            label_num += 1

        label = label_dic.get(label1)

        data_x.append(text)
        data_y.append(label)
    assert len(data_x) == len(data_y)
    return data_x, np.array(data_y).astype(int),apps


def svm_train():
    # train_x, train_y,apps = get_data_set(train_path)
    # test_x, test_y,apps = get_data_set(test_path)
    # pred_x,_,apps=get_data_set(pred_path)
    train_x, train_y, test_x, test_y, pred_x, apps, label_dic = load_data()
    # with open(CHANNEL_MODEL + 'svm_label.pkl', 'wb') as f:
    #     pickle.dump(label_dic, f)

    logging.info('train {} test{}'.format(len(train_x), len(test_x)))
    t=time.time()

    estim = HyperoptEstimator(classifier=liblinear_svc('clf'),max_evals=2,
                              preprocessing=[
                                  tfidf('tfidf', min_df=10, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)],
                              algo=tpe.suggest, trial_timeout=1200,refit=False)

    estim.fit(train_x, train_y)

    print(estim.score(test_x, test_y))
    # <<show score here>>
    print(estim.best_model())
    # <<show model here>>



def svm_pred():
    logging.info('pred')
    test_x, test_y = get_data_set(pred_path)
    with open(project_path + 'tfidf.pkl', 'rb') as f:
        vec = pickle.load(f)

    test_term_doc = vec.transform(test_x)

    with open(project_path + 'svm_model.pkl', 'rb') as f:
        lin_clf = pickle.load(f)

    test_preds = lin_clf.predict(test_term_doc)

    from sklearn.metrics import confusion_matrix, classification_report

    logging.info('\n {}'.format(classification_report(test_y, test_preds)))

if __name__ == "__main__":
    svm_train()
    #svm_pred()

