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
label_dic2 = {}
label1_label2={}
from sklearn_hierarchical_classification.constants import ROOT
label1_label2[ROOT]=[]
label_num = 0
label_num2=0
t = time.time()

import json
import math
from sklearn.calibration import CalibratedClassifierCV

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
def get_data_set(flie):
    global label_num
    global label_num2
    with open(flie) as f:
        lines = f.readlines()
    data_x = []
    data_y = []
    data_y2=[]
    apps=[]
    for li in lines:
        li=json.loads(li)
        text=li.get("jieba")
        labels=li.get("label_name","no123456") #label_1st
        if "#" in labels:
            label1=labels.split("#")[0]
        else:
            label1=labels[:4]
        label2=labels
        app=li.get("app")
        apps.append(app)

        if label1 not in label1_label2[ROOT]:
            label1_label2[ROOT].append(label1)
            label1_label2[label1]=[]

        if label2 not in label1_label2[label1]:
            label1_label2[label1].append(label2)

        # label = label_dic.get(label1)
        # label2 = label_dic2.get(label2)
        data_x.append(text)
        data_y.append(label2)
        # if "test" not in flie or "train" not in flie:
        #     data_x.append(text)
        #     data_y.append(label1)
        data_y2.append(label2)
    assert len(data_x) == len(data_y)
    return data_x, np.array(data_y).astype(str),apps,np.array(data_y2).astype(str)


def svm_train():
    train_x, train_y,apps,train_y2 = get_data_set(train_path)
    test_x, test_y,apps,test_y2 = get_data_set(test_path)
    pred_x,_,apps,_=get_data_set(pred_path)
    # with open(CHANNEL_MODEL + 'svm_label.pkl', 'wb') as f:
    #     pickle.dump(label_dic, f)

    logging.info('train {} test{}'.format(len(train_x), len(test_x)))
    t=time.time()
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
    print(label1_label2)
    time.sleep(20)
    tfidf_time = time.time()
    logging.info('time spend {}'.format(tfidf_time - t))

    logging.info('begin svm ')
    lin_clf = svm.LinearSVC(C=1)
    lin_clf = CalibratedClassifierCV(lin_clf)

    clf = HierarchicalClassifier(
        base_estimator=lin_clf,
        class_hierarchy=label1_label2,
    )

    clf.fit(trn_term_doc, train_y)

    print(clf.classes_)

    logging.info('end  svm ')
    # with open(project_path + 'svm_model.pkl', 'wb') as f:
    #     pickle.dump(lin_clf, f)

    train_preds = clf.predict(trn_term_doc)
    train_preds_prob = clf.predict_proba(trn_term_doc)
    print(len((clf.classes_)),train_preds_prob.shape)
    time.sleep(20)
    for reg,prob in zip(train_preds,train_preds_prob):
        print(reg,list(prob.argsort()[-1:][::-1]))
    time.sleep(20)
    from sklearn.metrics import classification_report

    logging.info('train {} accuracy_score {},  \n {}'.format('train',accuracy_score(train_y, train_preds),classification_report(train_y, train_preds)))
    t2 = time.time()
    logging.info('time spend {}'.format(t2 - t))

    test_term_doc = vec.transform(test_x)
    test_preds_1 = clf.predict_proba(test_term_doc)
    test_preds = clf.predict(test_term_doc)
    logging.info('train {} accuracy_score {},  \n {}'.format('train',accuracy_score(train_y, train_preds),classification_report(test_y2, test_preds)))

    dic_lab={}
    for k,v in label_dic2.items():
        dic_lab[v]=k
    test_preds = []
    test_preds=[]
    for prob in test_preds_1:
        test_preds.append(list(prob.argsort()[-2:][::-1]))


    test_y_name=[]
    test_preds_name=[]
    for  real, pred in zip(test_y2, test_preds):
        prd=pred[0]
        print(real, pred)
        for pr in pred:
            if real==clf.classes_[pr]:
                prd=pr
        test_y_name.append(real)
        test_preds_name.append(clf.classes_[prd])


    logging.info('{} model on {} data accuracy_score {} top2 test\n {}'.format("train", test_path,accuracy_score(test_y_name, test_preds_name),
                                                                classification_report(test_y_name, test_preds_name)))
    #cnf=classification_report(test_y_name, test_preds_name)

    # pred_term_doc = vec.transform(pred_x)
    # pred_preds_prob = lin_clf.predict_proba(pred_term_doc)
    # pred_preds=[]
    #
    # for prob in pred_preds_prob:
    #     pred_preds.append(list(prob.argsort()[-2:][::-1]))
    # with open(os.path.join(project_path,"hsvm.csv"),"w",encoding="utf8")  as f:
    #     f.writelines("id,label1,label2\n")
    #
    #     for ap,te in zip(apps,pred_preds):
    #         res=[ap]
    #         for t in te:
    #            res.append(dic_lab[t])
    #         f.writelines(','.join(res)+'\n')

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
