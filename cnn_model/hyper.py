
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
from  cnn_model.main_label_classify import  main_class
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
    acc=main_class()
    loss=-acc
    logging.info("acc {}, on test  set is {} and top2 acc {},loss {} time {},params: \n{}".format(acc,acc2,acc3,loss,t2-t1,params))
    return {'loss': loss, 'status': STATUS_OK}

def optimize(
             #trials,
             random_state=SEED):

    hyper={}
    hyper["batch_size"]=hp.quniform('batch_size', 100, 700, 5)
    hyper["warmup_proportion"]=hp.uniform('warmup_proportion', 0.0, 0.6)
    hyper["learning_rate"]=hp.uniform('learning_rate', 0.0, 0.6)
    hyper["optimizer"]=hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    hyper["l2_reg_lambda"]=hp.loguniform("l2_reg_lambda", 0.1, 0.5)
    hyper["num_filters"]=hp.quniform('batch_size', 100, 700, 20)
    hyper["dropout_prob"]=hp.uniform('conv_dropout_proba', 0.0, 0.35),
    hyper["activation"]=hp.choice('activation', ['relu', 'elu'])
    hyper["f1"]=hp.choice('f1', [False, True])
    hyper["f2"]=hp.choice('f2', [False, True])
    hyper["f3"]=hp.choice('f3', [False, True])
    hyper["f4"]=hp.choice('f4', [False, True])
    hyper["f5"]=hp.choice('f5', [False, True])
    hyper["f6"]=hp.choice('f6', [False, True])
    hyper["f7"]=hp.choice('f7', [False, True])
    hyper["f8"]=hp.choice('f8', [False, True])

    # space = { "ngram_range":hp.quniform("ngram_range",1, 4, 1),
    #           "C":hp.loguniform("C", np.log(1e-5), np.log(1e5)),
    #           "min_df": hp.choice('min_df', np.arange(10, 50, dtype=int)),
    #           "max_df":hp.choice('max_df', np.array([0.9,0.92,0.95])),
    #           "intercept_scaling":hp.loguniform("intercept_scaling", np.log(1e-1), np.log(1e1))
    #           }
    best = fmin(score, hyper, algo=tpe.suggest,
                max_evals=250)
    return best

if __name__ == "__main__":
    best_hyperparams = optimize(
        # trials
    )
    logging.info("The best hyperparameters are: ", "\n")
    logging.info(best_hyperparams)
    #svm_pred()

