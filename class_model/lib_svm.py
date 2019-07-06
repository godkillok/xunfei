import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,HashingVectorizer
from sklearn import svm
import logging
import numpy as np
import time
import os
import pickle  # pickle模块
from sklearn.metrics import accuracy_score
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# 12813123
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
    for li in lines:
        li=json.loads(li)
        text=li.get("jieba")
        label1=li.get("label_1st","no")

        if label1 not in label_dic.keys():
            label_dic[label1] = label_num
            label_num += 1

        label = label_dic.get(label1)

        data_x.append(text)
        data_y.append(label)
    assert len(data_x) == len(data_y)
    return data_x, np.array(data_y).astype(int)


def svm_train():
    train_x, train_y = get_data_set(train_path)
    test_x, test_y = get_data_set(test_path)
    pred_x,_=get_data_set(pred_path)
    # with open(CHANNEL_MODEL + 'svm_label.pkl', 'wb') as f:
    #     pickle.dump(label_dic, f)

    logging.info('train {} test{}'.format(len(train_x), len(test_x)))

    data_set = train_x + test_x+pred_x
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=10, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    #vec=HashingVectorizer(ngram_range=(1, 3))
    vec.fit_transform(data_set)
    #
    with open(project_path + 'tfidf.pkl', 'wb') as f:
        pickle.dump(vec, f)
    # with open(CHANNEL_MODEL + 'tfidf.pkl', 'rb') as f:
    #     vec = pickle.load(f)


    trn_term_doc = vec.transform(train_x)

    tfidf_time = time.time()
    logging.info('time spend {}'.format(tfidf_time - t))

    logging.info('begin svm ')
    lin_clf = svm.LinearSVC(C=0.5)
    lin_clf = CalibratedClassifierCV(lin_clf)
    lin_clf.fit(trn_term_doc, train_y)
    logging.info('end  svm ')
    with open(project_path + 'svm_model.pkl', 'wb') as f:
        pickle.dump(lin_clf, f)

    train_preds = lin_clf.predict(trn_term_doc)


    from sklearn.metrics import classification_report

    logging.info('train {} model {},  \n {}'.format('train',accuracy_score(train_y, train_preds),classification_report(train_y, train_preds)))
    t2 = time.time()
    logging.info('time spend {}'.format(t2 - t))

    test_term_doc = vec.transform(test_x)
    test_preds = lin_clf.predict(test_term_doc)

    dic_lab={}
    for k,v in label_dic.items():
        dic_lab[v]=k

    #logging.info('test \n {}'.format(classification_report(test_y, test_preds)))
    # with open(result,"w",encoding="utf8") as f: #1
    #     for id,real,pred in zip(test_id,test_y, test_preds):
    #         f.writelines(id+'\t'+dic_lab[real]+'\t'+dic_lab[pred]+'\n')
    #
    #
    # with open(result,"w",encoding="utf8") as f:
    #     for id,real,pred in zip(test_id,test_y, test_preds):
    #         if real!=pred:
    #             f.writelines(id+'\t'+dic_lab[real]+'\t'+dic_lab[pred]+'\n')

    test_y_name=[]
    test_preds_name=[]
    for  real, pred in zip( test_y, test_preds):
        test_y_name.append(dic_lab[real])
        test_preds_name.append(dic_lab[pred])

    logging.info('{} model on {} data{} set test\n {}'.format("train", test_path,accuracy_score(test_y_name, test_preds_name),
                                                                classification_report(test_y_name, test_preds_name)))
    cnf=classification_report(test_y_name, test_preds_name)

    test_preds_prob = lin_clf.predict_proba(test_term_doc)
    test_preds=[]
    for prob in test_preds_prob:
        test_preds.append(list(prob.argsort()[-2:][::-1]))

    test_y_name=[]
    test_preds_name=[]
    for  real, pred in zip( test_y, test_preds):
        prd=pred[0]
        #print(real, pred)
        for pr in pred:

            if real==pr:
                prd=real
        test_y_name.append(dic_lab[real])
        test_preds_name.append(dic_lab[prd])

    logging.info('{} model on {} data {} top2 test\n {}'.format("train", test_path,accuracy_score(test_y_name, test_preds_name),
                                                                classification_report(test_y_name, test_preds_name)))
    cnf=classification_report(test_y_name, test_preds_name)


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
