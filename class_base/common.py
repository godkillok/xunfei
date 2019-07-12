# -*- coding: utf-8 -*-
import os
project_path="/data/tanggp/xun_class//aichallenge"
train_path=os.path.join(project_path,"apptype_train.dat")
test_path=os.path.join(project_path,"app_desc.dat")
id_path=os.path.join(project_path,"apptype_id_name.txt")
import  jieba
import json
import re
from collections import defaultdict
import time
import urllib.request
token_url = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s"
# 1.获取token
api_key = 'hGs3TEt3sN3XcI3VyIAyuTQp'
api_secert = 'P7tCqnwMBEPs6bpEa4TOr4voTtAtTdxQ'

token_url = token_url % (api_key, api_secert)

r_str = urllib.request.urlopen(token_url).read()
r_str = str(r_str, encoding="utf-8")
token_data = json.loads(r_str)
token_str = token_data['access_token']

url_all = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/lexer?access_token=' + str(token_str)
fs=['ai_challenger_oqmrc_trainingset.json','ai_challenger_oqmrc_testa.json','ai_challenger_oqmrc_validationset.json']

def  segm(word1):

    data2 = {'text': word1}
    post_data = json.dumps(data2)
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                   'Content-Type': 'application/json'}
    req = urllib.request.Request(url=url_all, data=bytes(post_data, encoding="gbk"), headers=header_dict)
    res = urllib.request.urlopen(req).read()
    # print(res)
    r_data = str(res, encoding="GBK")
    # print(r_data)
    res = json.loads(r_data)

    seg = []
    for i in res.get('items'):
        seg.append(i.get('item'))
    sentence = ' '.join(seg)
    return str(sentence)

def  segment(text):
    retry=3
    while retry > 0:
        try:
            text= segm(text)
            retry = -1
        except Exception as e:
            retry -= 1
            print(e)
            time.sleep(1)
    return text


id_dic={}
with open(id_path,"r",encoding="utf8") as f:
    lines=f.readlines()
    for li in lines:
        li=li.strip()
        lid,lname=li.split()
        id_dic[lid]=lname
        jieba.add_word(lname)

for path in [train_path,test_path]:
    result=[]
    lab_count=defaultdict(int)
    with open(path,"r",encoding="utf8") as f:
        lines=f.readlines()
        for i,li  in enumerate(lines):
            if i%1000==0:
                print(i)
            li=li.strip()
            li=li.split()
            res={}
            res["app"]=li[0]
            if "train" in path:
                text= li[2]
            else:
                text= li[1]
            punctuation = r"""1234567890!"#$%&()*+,-./:;<=>?@[\]^_`{|}~。，"""

            res["text"]=text
            text2 = re.sub(r'[{}]+'.format(punctuation), ' ', str(text))
            seg_list=jieba.cut(text2.lower(), cut_all=False)
            res["jieba"]=' '.join(seg_list)
            res["baidu"]=segment(text)
            if "train" in path:
                llid=li[1]
                if len(llid.split("|"))>2:
                    print(llid)
                for i  in  range(len(llid.split("|"))):
                    lid=llid.split("|")[i]
                    res["label"] = lid
                    res["label_name"] = id_dic[lid[:4]]+'#'+id_dic[lid]
                    res["label_1st"] = id_dic[lid[:4]]
                    res["label_2st"] = id_dic[lid]
                    lab_count[res["label_2st"]]+=1
                    result.append(res)
            else:
                result.append(res)
    print(lab_count)
    if "train" not in path:
        with open(path.replace(".dat",".jieba_json"), "w", encoding="utf8") as f:
           for  res in result:
               f.writelines(json.dumps(res,ensure_ascii=False)+'\n')
    else:
        import random

        random.seed(10)
        random.shuffle(result)
        for k,v in lab_count.items():
            lab_count[k]=int(v*0.8)+1
        print(lab_count)
        with open(path.replace(".dat",".train_jieba_json"), "w", encoding="utf8") as f,  open(path.replace(".dat",".test_jieba_json"), "w", encoding="utf8") as f1:
           for  res in result:
               if lab_count[res["label_2st"]]>0:
                   lab_count[res["label_2st"]]-=1
                   f.writelines(json.dumps(res,ensure_ascii=False)+'\n')
               else:
                    f1.writelines(json.dumps(res,ensure_ascii=False)+'\n')