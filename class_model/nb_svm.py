from sklearn.base import BaseEstimator, ClassifierMixin
import os
import sys
currentUrl = os.path.dirname(__file__)
most_parenturl = os.path.abspath(os.path.join(currentUrl, os.pardir))
m_p, m_c = os.path.split(most_parenturl)
while 'xunfei' not in m_c:
    m_p, m_c = os.path.split(m_p)
sys.path.append(os.path.join(m_p, m_c))
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy import sparse
from class_model.load_data import load_data
from cnn_model.post_pred import post_eval
from sklearn.metrics import log_loss
import os
import sys
import pickle
import logging



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,HashingVectorizer
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        # y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, solver='sag',n_jobs=self.n_jobs).fit(x_nb, y)
        return self

# model = NbSvmClassifier(C=4, dual=True, n_jobs=-1)

model = NbSvmClassifier(C=4, dual=False, n_jobs=-1)
train_x, train_y, test_x, test_y, pred_x, apps, label_dic = load_data()
data_set=train_x+test_x+pred_x
# vec = TfidfVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.95,
#                       use_idf=1, smooth_idf=1, sublinear_tf=1)


#vec.fit_transform(data_set)
project_path="/data/tanggp/xun_class/aichallenge/"
# with open(project_path + 'tfidf.pkl', 'wb') as f:
#     pickle.dump(vec, f)
with open(project_path + 'tfidf.pkl', 'rb') as f:
    vec = pickle.load(f)


trn_term_doc = vec.transform(train_x)
test_term_doc = vec.transform(test_x)
train_loss = []
valid_loss = []
preds_train = np.zeros((len(train_x), len(train_y)))
preds_valid = np.zeros((len(test_x), len(test_y)))

TARGET_COLS=label_dic.values()
for i, j in enumerate(TARGET_COLS):
    print('Class:= '+str(j))
    y_train = np.where(train_y ==j, 1, 0)
    y_valid= np.where(test_y ==j, 1, 0)
    model.fit(trn_term_doc,y_train)
    preds_valid[:,i] = model.predict_proba(test_term_doc)[:,1]
    preds_train[:,i] = model.predict_proba(trn_term_doc)[:,1]
    train_loss_class=log_loss(y_train,preds_train[:,i])
    try:
        valid_loss_class=log_loss(y_valid,preds_valid[:,i])
    except  Exception as e:
        print(e)
        valid_loss_class= np.mean(train_loss)
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))
print("shape preds_valid {}".format(preds_valid.shape))
acc2=post_eval(preds_valid,test_y)
print("shape preds_valid top 2 {}".format(acc2))