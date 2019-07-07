from data_prepare import *
import tensorflow as tf
from pathlib import Path
from tensorflow.contrib import predictor
#
# def del_all_flags(FLAGS):
#     flags_dict = FLAGS._flags()
#     keys_list = [keys for keys in flags_dict]
#     for keys in keys_list:
#         FLAGS.__delattr__(keys)
#
# del_all_flags(tf.flags.FLAGS)
# from run_classify import *

import numpy as np

export_dir = os.path.join(FLAGS.output_dir, 'saved_model')
subd = [rot for rot, dr, _ in os.walk(export_dir) if 'variables' not in rot and 'temp' not in rot]
latest = str(sorted(list(set(subd)))[-1])
predict_fn = predictor.from_saved_model(latest)


def per_example_clean(item):
    title = item.get('title', '')
    tags = sorted(item.get('tags', []))
    pic_tags = [pic.split('\x02')[0] for pic in item.get('pic_tags', '').split('\x03')]
    pic_tags = sorted(pic_tags)
    source_user = item.get('source_user', '')
    senc = title + ' ' + ' ' + ' '.join(tags) + ' ' + ' '.join(pic_tags) + source_user
    senc = senc.replace('\n', '').replace('\r', '')
    if len(senc.split()) < 10:
        print('{}----{}'.format(item.get('id', ''), senc))
    lab = item['label']
    vid = item.get('id', '')
    com = ".jpeg_https~tfhub.dev~google~imagenet~inception_v3~feature_vector~1.txt"

    pic_lines='0,0'

    # lines.append()
    return [senc, str(lab), pic_lines]


def predict_online(line):
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    label = tokenization.convert_to_unicode(
        str(line[1]))  # this should compatible with format you defined in processor.
    text_a = tokenization.convert_to_unicode(line[0])
    text_b = None
    image = [float(fg) for fg in line[2].split(',')]
    example = InputExample(guid=0, text_a=text_a, text_b=text_b, label=label,image=image)
    feature = convert_single_example(0, example, label_list, FLAGS.max_seq_length, tokenizer)
    input_ids = np.reshape([feature.input_ids], (1, FLAGS.max_seq_length))
    input_mask = np.reshape([feature.input_mask], (1, FLAGS.max_seq_length))
    segment_ids = np.reshape([feature.segment_ids], (1, FLAGS.max_seq_length))
    image = np.reshape([feature.image], (1, 2))
    label_ids = [feature.label_id]
    feed_dict = {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids,
                 "label_ids": label_ids,"image":image}

    predictions = predict_fn(feed_dict)
    label_index = np.argmax(predictions)
    label_predict = index2label[label_index]
    # print("text_a:", line[0], "label_predict:", label_predict, ";possibility:", predictions)
    return label_predict, predictions


def create_pred_():
    import json
    #133
    file_name = '/data/tanggp/nsfw_input/nsfw_info_pred'
    with open(file_name, 'r', encoding='utf8') as f:
        items = f.readlines()
    result = []
    with open('nsfw_reslut','w',encoding='utf8') as f:
        for i,it in enumerate(items):
            it = json.loads(it)
            example = per_example_clean(it)
            if example is not None:
                vid=it.get('id','')
                label=it.get('label','')
                title=it.get('title','')
                url=it.get('img',{}).get('original_url','')
                google = it.get('special_tags', {}).get('erotic_google', 6)
                manual = it.get('special_tags', {}).get('erotic_manual', 6)
                f.writelines("{}\t {}\t{} \t{} \t{} \t{}\t {} \n".format(vid,label,google,manual,url,predict_online(example),title))
    return result


if __name__ == "__main__":
    processors = {
        "category": CategoryProcessor,
        "nsfw": NSFWProcessor
    }
    task_name = FLAGS.task_name.lower()
    processor = processors[task_name]()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    label_list = processor.get_labels()

    index2label = {i: label_list[i] for i in range(len(label_list))}
    print(index2label)
    num_labels = len(label_list)
    example = [
        '\u5165\u804c\u4e00\u5e74\u534a\u672a\u7b7e\u52b3\u52a8\u5408\u540c\u5c0f\u83f2\u6bd5\u4e1a\u4e8e\u67d0\u62a4\u6821\uff0c\u548c\u5176\u4ed6\u7684\u9ad8\u6821\u6bd5\u4e1a\u751f\u4e00\u6837\uff0c\u5979\u4e5f\u5f00\u59cb\u7740\u624b\u627e\u5de5\u4f5c\u3002\u5f88\u5feb\uff0c\u4e00\u5bb6\u6c11\u529e\u533b\u9662\u901a\u8fc7\u67d0\u62db\u8058\u7f51\u7ad9\u627e\u5230\u5c0f\u83f2\uff0c\u901a\u8fc7\u9762\u8bd5\u540e\uff0c\u5c0f\u83f2\u4fbf\u5f00\u59cb\u4e86\u81ea\u5df1\u7684\u804c\u573a\u751f\u6daf\u3002\u8f6c\u773c\u6bd5\u4e1a\u5de5\u4f5c\u8fd1\u4e00\u5e74\uff0c\u533b\u9662\u4ecd\u8fdf\u8fdf\u4e0d\u4e0e\u5176\u7b7e\u8ba2\u52b3\u52a8\u5408\u540c\uff0c\u5c0f\u83f2\u4e0e\u5355\u4f4d\u591a\u6b21\u6c9f\u901a\u534f\u5546\u672a\u679c\uff0c\u65e0\u5948\u5c06\u533b\u9662\u8bc9\u81f3\u6cd5\u9662\u000d\u000a\u652f\u4ed8\u5de5\u8d44',
        '0']

    #predict_online(example)
    create_pred_()
