# -*- coding: utf-8 -*-
import os
project_path="/data/tanggp/xfyun/classify/aichallenge"
add_path=os.path.join(project_path,"add_data.csv")
test_path=os.path.join(project_path,"app_desc.jieba_json")
id_path=os.path.join(project_path,"apptype_id_name.txt")
train_path=os.path.join(project_path,"new_train.jieba_json")
import  jieba
import json
import re
from collections import defaultdict
import time
import urllib.request

id_dic={}
add_dic={}
with open(id_path,"r",encoding="utf8") as f:
    lines=f.readlines()
    for li in lines:
        li=li.strip()
        lid,lname=li.split()
        id_dic[lid]=lname

with open(add_path,"r",encoding="utf8") as f:
    lines=f.readlines()
    for li in lines:
        li=li.strip()
        lid,label=li.split(',')
        add_dic[lid]=label

for path in [test_path]:
    result=[]
    lab_count=defaultdict(int)
    with open(path,"r",encoding="utf8") as f:
        lines=f.readlines()
        for li in lines:
            li=json.loads(li)
            app=li.get("app")
            if app in add_dic:
                lid=add_dic[app]
                li["label"] = lid
                li["label_name"] = id_dic[lid[:4]] + '#' + id_dic[lid]
                li["label_1st"] = id_dic[lid[:4]]
                li["label_2st"] = id_dic[lid]
                lab_count[li["label_2st"]] += 1
                result.append(li)
with open(train_path, "w", encoding="utf8") as f:
   for  res in result:
       f.writelines(json.dumps(res,ensure_ascii=False)+'\n')
