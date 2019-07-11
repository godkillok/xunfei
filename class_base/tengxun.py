import os
import json
project_path="/data/tanggp/xfyun/classify/aichallenge"
test_path=os.path.join(project_path,"apptype_train.test_jieba_json")
train_path=os.path.join(project_path,"apptype_train.train_jieba_json1")
pred_path=os.path.join(project_path,"app_desc.jieba_json")
embed_path=os.path.join(project_path,"tencent")
pre_train=os.path.join("/data/tanggp","Tencent_AILab_ChineseEmbedding.txt")
word_list=[]
for fi in [test_path,train_path,pred_path]:
    with open(fi,"r",encoding="utf8") as f:
        lines=f.readlines()
        for li in lines:
            li=json.loads(li)
            jieba=li.get("jieba")
            word_list+=jieba.split()

word_list=set(word_list)
print(word_list[:10])
embed=[]
with open(pre_train,"r",encoding="utf8") as f:
    for li in f.readline():
        word=li.split()[0]
        if word in word_list:
            embed.append(li)
with open(embed_path,"w",encoding="utf8") as f:
    for em in embed:
        f.writelines(em)