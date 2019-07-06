import os
project_path="/data/tanggp/xfyun/classify/aichallenge"
train_path=os.path.join(project_path,"apptype_train.dat")
test_path=os.path.join(project_path,"app_desc.dat")
id_path=os.path.join(project_path,"apptype_id_name.txt")
import  jieba
import json
import re
from collections import defaultdict
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
        for li  in lines:
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