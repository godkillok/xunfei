#!/usr/bin/env python
import tensorflow as tf
import os
import re
import numpy as np
import random

import random
import json
import time
flags = tf.app.flags
path="/data/tanggp/xun_class//aichallenge"
flags.DEFINE_string("data_dir", path, "Directory containing the dataset")
flags.DEFINE_string("pad_word", '<pad>', "used for pad sentence")
flags.DEFINE_string("OOV", '<unk>', "used for pad sentence")
flags.DEFINE_string("path_vocab", os.path.join(path,"textcnn_words.txt"), "used for word index")
flags.DEFINE_string("path_author",  os.path.join(path, 'textcnn_author_sort'), "Directory containing the dataset")
flags.DEFINE_string("path_label",  os.path.join(path, 'textcnn_label_sort'), "Directory containing the dataset")
FLAGS = flags.FLAGS

sentence_max_len = 250
pad_word = FLAGS.pad_word
OOV =FLAGS.OOV
label_class=[]
author_calss=[]
def feature_auto(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, list):
        if isinstance(value[0],int):
            try:
                tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            except:
                print(value)
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        elif isinstance(value[0],float):
            try:
                tf.train.Feature(int64_list=tf.train.FloatList(value=value))
            except:
                print(value)
            return tf.train.Feature(int64_list=tf.train.FloatList(value=value))
        else:
            print("list type error")

    elif isinstance(value, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    elif isinstance(value, float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def parse_line_dict(record,vocab_dict,author_dict,label_dict):
    record = json.loads(record)
    tokens=record.get("jieba")
    tokens=tokens.split()
    text = [vocab_dict.get(r,OOV) for r in tokens]
    app=record.get("app")
    if "label"  in record:
        label=record["label"]
        if "#" in label:
            author = label.split("#")
        else:
            author = label[:4]

        au=author_dict[author]
        return [text, label_dict.get(label),au,app]
    else:
        return [text, None, None,app]


def per_thouds_lines_dict(result_lines, path_text, count,flag_name=''):
    tf_lines = []


    for rl_num,rl in enumerate(result_lines):
        text=rl[0]
        label=rl[1]
        author=rl[2]
        app=rl[3]
        if len(text) >= sentence_max_len:
            text = text[0: sentence_max_len]
        else:
            text += [pad_word] * (sentence_max_len - len(text))
        g={"text":text,"label":label,"author":author,"app":app}
        tf_lines.append(g)
    write_tfrecords(tf_lines, path_text, count)

def ini():
    global pad_word,OOV

    with open(FLAGS.path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab_dict = {l.strip(): (i) for i, l in enumerate(lines)}
        pad_word=vocab_dict.get(pad_word)
        OOV=vocab_dict.get(OOV)
        print("pad_word {},OOV {}".format(pad_word,OOV))


    with open(FLAGS.path_label, 'r', encoding='utf8') as f:
        lines = f.readlines()
        label_dict = {l.strip().split("\x01\t")[0]: i for i, l in enumerate(lines)}


    with open(FLAGS.path_author, 'r', encoding='utf8') as f:
        lines = f.readlines()
        i=0
        author_dict={}
        for  l in lines:
            if int(l.strip().split("\x01\t")[1]) > 10:
                author_dict[l.strip().split("\x01\t")[0]]=i
                i+=1

    return vocab_dict,author_dict,label_dict

def generate_tf_dic(path_text,vocab_dict,author_dict,label_dict):


    result_lines = []
    count = 0
    with open(path_text, 'r', encoding='utf8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            count+=1
            result_lines.append(parse_line_dict(line,vocab_dict,author_dict,label_dict))
            if count>0 and count % 50000 == 0:
                print(count)
                per_thouds_lines_dict(result_lines, path_text, count)
                result_lines = []
        if len(result_lines)>0:
            per_thouds_lines_dict(result_lines, path_text, count)


def write_tfrecords(tf_lines, path_text, count):
    (root_path, output_filename) = os.path.split(path_text)
    output_filename = output_filename.split('.')[0]
    output_filename='author_text_cnn_'+output_filename
    output_file = output_filename + '_' + str(count)+ '.tfrecords'

    print("Start to convert {} to {}".format(len(tf_lines), os.path.join(root_path, output_file)))

    writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
    random.shuffle(tf_lines)
    num = 0
    for i,data in enumerate(tf_lines):
        if i==0:
            print("Start to convert {}".format(data))
        text = data["text"]
        label = data["label"]
        author=data["author"]
        app=data["app"]
        if not label is  None:
            if i == 0:
                print("OK {}".format(i))
            example = tf.train.Example(features=tf.train.Features(feature={
                'text': feature_auto(list(text)),
                'label': feature_auto(int(label)),
                'author': feature_auto(int(author)),
                "app":feature_auto(app),
            }))
        else:
            if i == 0:
                print("NOT OK {}".format(i))
            example = tf.train.Example(features=tf.train.Features(feature={
                'text': feature_auto(list(text)),
                "app": feature_auto(app)
            }))

        writer.write(example.SerializeToString())
        num += 1
        # if num % 1000 == 0:
        #     output_file = output_filename + '_' + str(count) + '_' + str(num)+'_' + flag_name + '.tfrecords'
        #     writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
        #     print("Start convert to {}".format(output_file))


def main():
    vocab_dict, author_dict, label_dict= ini()
    generate_tf_dic(os.path.join(FLAGS.data_dir, 'apptype_train.train_jieba_json1'),vocab_dict,author_dict,label_dict)
    generate_tf_dic(os.path.join(FLAGS.data_dir, 'apptype_train.test_jieba_json'),vocab_dict,author_dict,label_dict)
    generate_tf_dic(os.path.join(FLAGS.data_dir, 'app_desc.jieba_json'), vocab_dict, author_dict, label_dict)
    # s3_input = FLAGS.data_dir
    # for root, dirs, files in os.walk(s3_input):
    #     for file in files:
    #         # if file.endswith("ain_set.csv"):
    #         #     print('start to process file {}'.format(file))
    #         generate_tf_dic(os.path.join(root, file))
    # os.system('cd {}'.format(s3_input))
    # os.system('find . -name "*" -type f -size 0c | xargs -n 1 rm -f')


if __name__ == "__main__":
    main()
