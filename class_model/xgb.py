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
# import seaborn as sns
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
dataset = train_x + test_x + pred_x
tfidf = TfidfVectorizer(min_df=5, max_df=0.9, use_idf=1, smooth_idf=1, ngram_range=(1, 3),
                        strip_accents='unicode',
                        lowercase=True, analyzer='word', token_pattern=r'\w+', sublinear_tf=True,
                        stop_words='english')

cvec = CountVectorizer(min_df=5, ngram_range=(1, 3), max_features=maxFeats,
                       strip_accents='unicode',
                       lowercase=True, analyzer='word', token_pattern=r'\w+',
                       stop_words='english')
cvec.fit(dataset)

tfidf.fit(dataset)
svdT = TruncatedSVD(n_components=390)
svdTFit = svdT.fit(tfidf.transform(dataset))


def buildFeats(x_text):
    temp = {}
    temp['doc_len'] = [len(x.split()) for x in x_text]
    temp['unique_words'] = [len(set(x.split())) for x in x_text]

    temp_tfidf = tfidf.transform(x_text)
    temp['tfidf_sum'] = temp_tfidf.sum(axis=1)
    temp['tfidf_mean'] = temp_tfidf.mean(axis=1)
    temp['tfidf_len'] = (temp_tfidf != 0).sum(axis=1)

    temp_cvec = cvec.transform(texts)
    temp['cvec_sum'] = temp_cvec.sum(axis=1)
    temp['cvec_mean'] = temp_cvec.mean(axis=1)
    temp['cvec_len'] = (temp_cvec != 0).sum(axis=1)

    tempc = list(temp.columns)
    temp_lsa = svdT.transform(temp_tfidf)

    for i in range(np.shape(temp_lsa)[1]):
        tempc.append('lsa' + str(i + 1))
    temp = pd.concat([temp, pd.DataFrame(temp_lsa, index=temp.index)], axis=1)

    return temp, tempc
trainDf, traincol = buildFeats(train_x)
testDf, testcol = buildFeats(test_x)
predDf, predcol = buildFeats(pred_x)

trainDf.columns = traincol
testDf.columns = testcol
predDf.columns = predcol
lr_clf = LogisticRegression(random_state=0, solver='sag',multi_class='ovr', verbose = 1)

lr_clf.fit(trainDf, y_train)

train_preds = lin_clf.predict(trainDf)

from sklearn.metrics import classification_report

logging.info('train {} accuracy_score {},  \n {}'.format('train', accuracy_score(train_y, train_preds),
                                                         classification_report(train_y, train_preds)))


test_preds = lin_clf.predict(testDf)

logging.info('train {} accuracy_score {},  \n {}'.format('test', accuracy_score(test_y, test_preds),
                                                         classification_report(test_y, test_preds)))