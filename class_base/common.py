import os
project_path="/data/tanggp/xfyun/classify/aichallenge"
train_path=os.path.join(project_path,"apptype_train.dat")
test_path=os.path.join(project_path,"app_desc.dat")
id_path=os.path.join(project_path,"apptype_id_name.txt")
import  jieba
import json
id_dic={}
with open(id_path,"r",encoding="utf8") as f:
    lines=f.readlines()
    for li in lines:
        li=li.strip()
        lid,lname=li.split()
        id_dic[lid]=lname

for path in [train_path,test_path]:
    result=[]
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
            res["text"]=text
            seg_list=jieba.cut(text.lower(), cut_all=False)
            res["jieba"]=' '.join(seg_list)
            if "train" in path:
                lid=li[1]
                if len(lid.split("|")) == 2:
                    print(lid.split("|"))
                lid=lid.split("|")[0]

                res["label"] = lid
                res["label_name"] = id_dic[lid[:4]]+'#'+id_dic[lid]
                res["label_1st"] = id_dic[lid[:4]]
                res["label_2st"] = id_dic[lid]
            result.append(res)

    with open(path.replace(".dat",".jieba_json"), "w", encoding="utf8") as f:
       for  res in result:
           f.writelines(json.dumps(res)+'\n')