from sklearn.decomposition import TruncatedSVD
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# from nltk.corpus import stopwords
import re
import gc
# import seaborn as sns1
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import os
import sys
import pickle
import logging

currentUrl = os.path.dirname(__file__)
most_parenturl = os.path.abspath(os.path.join(currentUrl, os.pardir))
m_p, m_c = os.path.split(most_parenturl)
while 'xunfei' not in m_c:
    m_p, m_c = os.path.split(m_p)
sys.path.append(os.path.join(m_p, m_c))
project_path = "/data/tanggp/xfyun/classify/aichallenge"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from class_model.load_data import load_data
from sklearn.calibration import CalibratedClassifierCV

train_x, train_y, test_x, test_y, pred_x, apps, label_dic = load_data()
if "no" in label_dic:
    del label_dic["no"]
# dataset = train_x + test_x + pred_x
# tfidf = TfidfVectorizer(min_df=5, max_df=0.9, use_idf=1, smooth_idf=1, ngram_range=(1, 3),
#                         strip_accents='unicode',
#                         lowercase=True, analyzer='word', token_pattern=r'\w+', sublinear_tf=True,
#                         stop_words='english')
#
# cvec = CountVectorizer(min_df=5, ngram_range=(1, 3), max_df=0.9,
#                        strip_accents='unicode',
#                        lowercase=True, analyzer='word', token_pattern=r'\w+',
#                        stop_words='english')
# cvec.fit(dataset)
#
# tfidf.fit(dataset)
# svdT = TruncatedSVD(n_components=390)
# svdTFit = svdT.fit(tfidf.transform(dataset))
#
# with open(project_path + 'tfidf.pkl', 'wb') as f:
#     pickle.dump(tfidf, f)
# with open(project_path + 'svdTFit.pkl', 'wb') as f:
#     pickle.dump(svdTFit, f)
# with open(project_path + 'cvec.pkl', 'wb') as f:
#     pickle.dump(cvec, f)


with open(project_path + 'tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open(project_path + 'svdTFit.pkl', 'rb') as f:
    svdTFit = pickle.load(f)
with open(project_path + 'cvec.pkl', 'rb') as f:
    cvec = pickle.load(f)
logging.info("pre bulid feature....")

def buildFeats(x_text):
    temp = {}
    # temp['doc_len'] = [len(x.split()) for x in x_text]
    # temp['unique_words'] = [len(set(x.split())) for x in x_text]
    #
    temp_tfidf = tfidf.transform(x_text)
    # temp['tfidf_sum'] =list(temp_tfidf.sum(axis=1))
    # temp['tfidf_mean'] = list(temp_tfidf.mean(axis=1))
    # temp['tfidf_len'] = list((temp_tfidf != 0).sum(axis=1))
    #
    # temp_cvec = cvec.transform(x_text)
    # temp['cvec_sum'] = list(temp_cvec.sum(axis=1))
    # temp['cvec_mean'] = list(temp_cvec.mean(axis=1))
    # temp['cvec_len'] = list((temp_cvec != 0).sum(axis=1))

    tempc = list(temp.keys())
    temp_lsa = svdTFit.transform(temp_tfidf)

    for i in range(np.shape(temp_lsa)[1]):
        tempc.append('lsa' + str(i + 1))
    temp=pd.DataFrame.from_dict(temp)
    #temp = pd.concat([temp, pd.DataFrame(temp_lsa, index=temp.index)], axis=1)

    return temp_lsa, tempc

trainDf, traincol = buildFeats(train_x)
#print(trainDf.dtypes)
import time
time.sleep(30)
testDf, testcol = buildFeats(test_x)
predDf, predcol = buildFeats(pred_x)
logging.info("bulid feature done trainDf.shape {}".format(trainDf.shape))
# trainDf.columns = traincol
# testDf.columns = testcol
# predDf.columns = predcol


parms = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': len(label_dic),
    'metric': {'multi_logloss'},
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_iterations': 400,
    'num_leaves': 95,
    'min_data_in_leaf': 60,
    'lambda_l1': 1.0,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5}

rnds = 260
print('Format a Train and Validation Set for LGB')
# print(trainDf.describe())
d_train = lgb.Dataset(trainDf, label=train_y)
d_val = lgb.Dataset(testDf, label=test_y)

mod = lgb.train(parms, train_set=d_train, num_boost_round=rnds,
               valid_sets=[d_val], valid_names=['dval'], verbose_eval=20,
               early_stopping_rounds=20)

test_preds = mod.predict(d_val,axis=1)
logging.info('xgbt train {} accuracy_score {},  \n {}'.format('test', accuracy_score(test_y, test_preds),
                                                         classification_report(test_y, test_preds)))



lr_clf = LogisticRegression(random_state=0, solver='saga',multi_class='ovr', max_iter=1000,verbose =False ,n_jobs=-1)
#
lr_clf.fit(trainDf, train_y)
#
train_preds = lr_clf.predict(trainDf)

from sklearn.metrics import classification_report

logging.info('train {} accuracy_score {},  \n {}'.format('train', accuracy_score(train_y, train_preds),
                                                         classification_report(train_y, train_preds)))


test_preds = lr_clf.predict(testDf)

logging.info('train {} accuracy_score {},  \n {}'.format('test', accuracy_score(test_y, test_preds),
                                                         classification_report(test_y, test_preds)))
