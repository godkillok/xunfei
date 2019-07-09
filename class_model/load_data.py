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
from collections import defaultdict
import json
import random
from sklearn.calibration import CalibratedClassifierCV

def get_data_set(flie):
    global label_num
    with open(flie) as f:
        lines = f.readlines()
    lab_count=defaultdict(int)
    lines_json=[]
    for li in lines:
        li = json.loads(li)
        label1 = li.get("label", "no")
        lab_count[label1]+=1
        lines_json.append(li)

    need_repeat={}
    num=20
    for k,v in lab_count.items():
        if v<num:
            need_repeat[k]=int(num/v)+1

    if flie==train_path:
        for li in lines:
            li = json.loads(li)
            label1 = li.get("label", "no")
            i=0
            while i<need_repeat.get(label1,-1):
                lines_json.append(li)
                i+=1
    random.shuffle(lines_json)
    data_x = []
    data_y = []
    apps=[]
    for li in lines_json:

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


def load_data():
    train_x, train_y,apps = get_data_set(train_path)
    test_x, test_y,apps = get_data_set(test_path)
    pred_x,_,apps=get_data_set(pred_path)
    return train_x,train_y,test_x, test_y,pred_x,apps,label_dic