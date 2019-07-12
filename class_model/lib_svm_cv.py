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
project_path="/data/tanggp/xun_class//aichallenge/"
test_path=os.path.join(project_path,"apptype_train.test_jieba_json")
train_path=os.path.join(project_path,"apptype_train.train_jieba_json")
pred_path=os.path.join(project_path,"app_desc.jieba_json")
label_dic = {}
label_num = 0
t = time.time()

import json

from sklearn.calibration import CalibratedClassifierCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [0.1,1, 10, 100]},
                    {'kernel': ['linear'], 'C': [0.1,1, 10, 100]}]
scores = ['precision', 'recall']
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


def  cv():
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        train_x, train_y, apps = get_data_set(train_path)
        test_x, test_y, apps = get_data_set(test_path)
        pred_x, _, apps = get_data_set(pred_path)

        with open(project_path + 'tfidf.pkl', 'rb') as f:
            vec = pickle.load(f)

        trn_term_doc = vec.transform(train_x)

        # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
        clf = GridSearchCV(SVC(), tuned_parameters, cv=3,
                           scoring='%s_macro' % score)
        # 用训练集训练这个学习器 clf
        clf.fit(trn_term_doc, train_y)

        print("Best parameters set found on development set:")
        print()

        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print(clf.best_params_)

        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        # 看一下具体的参数间不同数值的组合后得到的分数是多少
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        trn_term_doc = vec.transform(test_x)
        y_true, y_pred = test_y, clf.predict(trn_term_doc)

        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))

        print()

def svm_train():
    train_x, train_y,apps = get_data_set(train_path)
    test_x, test_y,apps = get_data_set(test_path)
    pred_x,_,apps=get_data_set(pred_path)
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

    tfidf_time = time.time()
    logging.info('time spend {}'.format(tfidf_time - t))

    logging.info('begin svm ')
    lin_clf = svm.LinearSVC(C=0.1)
    lin_clf = CalibratedClassifierCV(lin_clf)
    lin_clf.fit(trn_term_doc, train_y)
    logging.info('end  svm ')
    with open(project_path + 'svm_model.pkl', 'wb') as f:
        pickle.dump(lin_clf, f)

    train_preds = lin_clf.predict(trn_term_doc)


    from sklearn.metrics import classification_report

    logging.info('train {} accuracy_score {},  \n {}'.format('train',accuracy_score(train_y, train_preds),classification_report(train_y, train_preds)))
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
    logging.info("\n"*3)

    logging.info('{} model on {} data accuracy_score {} set test\n {}'.format("test", test_path,accuracy_score(test_y_name, test_preds_name),
                                                                classification_report(test_y_name, test_preds_name)))
    # cnf=classification_report(test_y_name, test_preds_name)
    #
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
    if len(dic_lab)>30:
        logging.info('{} model on {} data accuracy_score {} top2 test\n {}'.format("train", test_path,accuracy_score(test_y_name, test_preds_name),
                                                                classification_report(test_y_name, test_preds_name)))
    cnf=classification_report(test_y_name, test_preds_name)

    pred_term_doc = vec.transform(pred_x)
    pred_preds_prob = lin_clf.predict_proba(pred_term_doc)
    pred_preds=[]

    for prob in pred_preds_prob:
        print(prob.sort()[-2:][::-1])
        pred_preds.append(list(prob.argsort()[-2:][::-1]))
    with open(os.path.join(project_path,"ag2.csv"),"w",encoding="utf8")  as f:
        f.writelines("id,label1,label2\n")

        for ap,te in zip(apps,pred_preds):
            res=[ap]
            for t in te:
               res.append(dic_lab[t])
            f.writelines(','.join(res)+'\n')

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
    cv()
    #svm_train()
    #svm_pred()
