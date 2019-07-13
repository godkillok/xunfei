#!/usr/bin/python3
# -*- coding: utf-8 -*-
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import os
import time
import pickle
import tensorflow as tf
from train_model import model_fn
import json
import logging
#from best_checkpoint_copier import BestCheckpointCopier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from logger import get_logger
log_file_name = os.path.basename(__file__).split('.', 1)[0] + '.log'
# Save params

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
# 当日志文件大小小于5M时，则以追加模式写
if os.path.exists(log_file_name) is False or os.path.getsize(log_file_name) / 1024 / 1024 < 5:
    logger = get_logger(log_file_name, mode='a')
else:
    # 否则删除以前的日志
    logger = get_logger(log_file_name)

flags = tf.app.flags
# configurations for training1
path="/data/tanggp/xun_class/aichallenge"
flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_string("data_dir",path , "The input datadir.")
flags.DEFINE_integer("shuffle_buffer_size", 20000, "dataset shuffle buffer size")  # 只影响取数据的随机性
flags.DEFINE_integer("num_parallel_calls", 40, "Num of cpu cores")
flags.DEFINE_integer("num_parallel_readers", 40, "Number of files read at the same time")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate")
flags.DEFINE_integer("steps_check", 500, "steps per checkpoint")
flags.DEFINE_string("train_file", os.path.join(path,"author_text_cnn_apptype_train_26320.tfrecords"), "train file pattern")
flags.DEFINE_string("valid_file", os.path.join(path,"author_text_cnn_apptype_train_6152.tfrecords"), "train file pattern")
#flags.DEFINE_string("valid_file", "/data/tanggp/youtube8m/text_cnn_txt_golden_*", "evalue file pattern")
# flags.DEFINE_string("emb_file", "/data/tanggp/xun_class/merge_sgns_bigram_char300.txt-1", "Path for pre_trained embedding")
flags.DEFINE_string("emb_file", "/data/tanggp/xun_class/cc.zh.300.bin", "Path for pre_trained embedding")

#flags.DEFINE_string("emb_file", "", "Path for pre_trained embedding")
flags.DEFINE_string("params_file", os.path.join(path,"textcnn_dataset_params.json"), "parameters file")
flags.DEFINE_string("word_path", os.path.join(path,"textcnn_words.txt"), "word vocabulary file")
flags.DEFINE_string("model_dir", os.path.join(path,"textcnn_model","base"), "Path to save model")
flags.DEFINE_string("result_file", os.path.join(path,"textcnn_model","base","base_result.txt"), "Path to save predict result")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for.")
# configurations for the model
flags.DEFINE_float("dropout_prob", 0.2, "Dropout rate")  # 以0.2的概率drop out
flags.DEFINE_integer("num_epoches", 100, "num of epoches")
flags.DEFINE_integer("early_stop_epoches", 5, "Stop train if eval result doesn't improve after 2 epoches")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-spearated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularization lambda (default: 0.0)")
flags.DEFINE_integer("word_dim", 200, "Embedding size for chars")
flags.DEFINE_integer("feature_dim", 16, "Embedding size for author feature")

FLAGS = tf.app.flags.FLAGS
assert 0 <= FLAGS.dropout_prob < 1, "dropout_prob rate between 0 and 1"
assert FLAGS.learning_rate > 0, "learning rate must larger than zero"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示warning 和 error
tf.logging.set_verbosity(logging.INFO)


timestamp = time.strftime("%m_%d_%H", time.localtime())
model_dir = os.path.join(FLAGS.model_dir)
# if FLAGS.do_train:
#
#     with open('model_dir.pkl', 'wb') as f:
#         pickle.dump(model_dir, f)
# else:
#     with open('model_dir.pkl', 'rb') as f:
#         model_dir = pickle.load(f)

def input_fn(filenames, config, shuffle_buffer_size):
    def parser(record):
        keys_to_features = {
            "text": tf.FixedLenFeature([config['max_length']], tf.int64),
            "author": tf.FixedLenFeature([1], tf.int64),
            "label": tf.FixedLenFeature([1], tf.int64)}
        parsed = tf.parse_single_example(record, keys_to_features)
        return {"text": parsed['text'], 'label':parsed['label'] ,"author": parsed['author']}

    # Load txt file, one example per line
    files = tf.data.Dataset.list_files(filenames)  # A dataset of all files matching a pattern.
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(FLAGS.num_epoches)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=parser, batch_size=FLAGS.batch_size,
                                                          num_parallel_calls=FLAGS.num_parallel_calls))
    return dataset
def id_word_map():
    with open(FLAGS.word_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab_dict = {str(i):l.strip() for i, l in enumerate(lines)}
    return vocab_dict

def top_2_label_code(test_preds_prob,test_y):
    test_preds = []
    for prob in test_preds_prob:
        test_preds.append(list(prob.argsort()[-2:][::-1]))

    test_y_name = []
    test_preds_code= []
    for real, pred in zip(test_y, test_preds):
        prd = pred[0]
        # print(real, pred)
        for pr in pred:
            if real == pr:
                prd = real
        test_y_name.append(real)
        test_preds_code.append(prd)
    return test_y_name,test_preds_code

def main_class_hyper(hyper):
    start = time.time()
    # Loads parameters from json file
    vocab_dict = id_word_map()
    # with open(FLAGS.params_file) as f:
    #     config = json.load(f)

    config={}
    config["train_size"]=26320
    config["max_length"] = 250
    config["id_word"] = vocab_dict
    config["word_dim"] = 300
    if config["train_size"] < FLAGS.shuffle_buffer_size:
        FLAGS.shuffle_buffer_size = config["train_size"]
    batch_size=hyper["batch_size"]
    num_epoches=hyper["batch_size"]
    warmup_proportion=hyper["warmup_proportion"]
    learning_rate=hyper["learning_rate"]
    l2_reg_lambda=hyper["l2_reg_lambda"]
    num_filters=hyper["num_filters"]
    dropout_prob=hyper["dropout_prob"]
    activation=hyper["activation"]
    f1,f2, f3, f4, f5, f6, f7, f8=hyper["f1"],hyper["f2"],hyper["f3"],hyper["f4"],hyper["f5"],hyper["f6"],hyper["f7"],hyper["f8"]
    filter_sizes=[]

    for i,fil in enumerate([f1,f2,f3,f4,f5,f6,f7,f8]):
        if fil:
            filter_sizes.append(str(i+1))
    if len(filter_sizes)==0:
        return 100
    filter_sizes=','.join(filter_sizes)
    model_dir = os.path.join(FLAGS.model_dir)
    train_steps = int(config["train_size"] / batch_size * num_epoches)
    logger.info('The number of training steps is {}'.format(train_steps))
    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=FLAGS.steps_check, session_config=session_config,
                                        keep_checkpoint_max=3,tf_random_seed=1)
    num_warmup_steps = int(train_steps *warmup_proportion)
    early_stop_steps = int(train_steps * 0.4)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            "id_word": config["id_word"],
            "train_size": config["train_size"],
            'max_length': config["max_length"],
            'emb_file': FLAGS.emb_file,
            'learning_rate': learning_rate,
            'l2_reg_lambda': l2_reg_lambda,
            'dropout_prob': dropout_prob,
            'word_dim': config["word_dim"],
            'vocab': FLAGS.word_path,
            "activation":activation,
            'num_filters': num_filters,
            'filter_sizes': list(map(int, filter_sizes.split(","))),
            'num_warmup_steps': num_warmup_steps,
            'train_steps': train_steps,
            'summary_dir': model_dir,
            "label_size": 125,
            'use_focal_loss': False,
            'use_author_feature': False,
            'use_category_feature': False,
            'use_keyword_feature': False,
            'feature_dim': FLAGS.feature_dim
        }
    )
    no_increase_steps = int(config["train_size"] / FLAGS.batch_size * FLAGS.early_stop_epoches)

    # 用于early stop
    # early_stop_hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, metric_name='f1',
    #                                                                 max_steps_without_increase=no_increase_steps,
    #                                                                min_steps=early_stop_steps, run_every_secs=120)
    acc2=0

    # timeline_hook = tf.train.ProfilerHook(save_steps=FLAGS.steps_check, output_dir=model_dir + '/timeline/')
    if FLAGS.do_train == True:
        input_fn_for_train = lambda: input_fn(FLAGS.train_file, config, FLAGS.shuffle_buffer_size)
        train_spec = tf.estimator.TrainSpec(input_fn=input_fn_for_train, max_steps=train_steps)
        input_fn_for_eval = lambda: input_fn(FLAGS.valid_file, config, 0)
        # best_copier = BestCheckpointCopier(name='best',  # directory within model directory to copy checkpoints to
        #         checkpoints_to_keep=1,  # number of checkpoints to keep
        #         score_metric='acc',  # metric to use to determine "best"
        #         compare_fn=lambda x, y: x.score > y.score,
        #         sort_reverse=True)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=12000)  # exporters=best_copier
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        logger.info("Switch to the current directory and Run the command line:" \
                    "tensorboard --logdir=%s" \
                    "\nThen open http://localhost:6006/ into your web browser" % timestamp)
        logger.info("after train and evaluate")
    if FLAGS.do_predict == True:
        best_dir = model_dir + '/best'

        path_label = os.path.join(FLAGS.data_dir, 'textcnn_label_sort')
        with open(path_label, 'r', encoding='utf8') as f:
            lines = f.readlines()
            id2label = {i: l.strip().split("\x01\t")[0] for i, l in enumerate(lines)}

        # predict
        predict_label_list = []
        true_label_list = []
        prob_list=[]
        true_label_code=[]
        input_fn_for_test = lambda: input_fn(FLAGS.valid_file, config, 0)
        output_results = estimator.predict(input_fn_for_test, checkpoint_path=tf.train.latest_checkpoint(best_dir))

        with open(FLAGS.result_file, 'w') as writer:
            for prediction in output_results:
                predict_label_id = prediction["predict_label_ids"]
                true_label_id = prediction["true_label_ids"]
                prob_list.append(prediction["probabilities"])
                predict_label = id2label[predict_label_id]
                true_label = id2label[true_label_id]
                predict_label_list.append(predict_label)
                true_label_code.append(true_label_id)
                true_label_list.append(true_label)
                writer.write(predict_label + '\t' + true_label + "\n")
        test_y_name, test_preds_code=top_2_label_code(prob_list, true_label_code)
        acc2=accuracy_score( test_y_name, test_preds_code)
        acc1 = accuracy_score(true_label_list, predict_label_list)
        logger.info(best_dir)
        #logger.info(classification_report(true_label_list, predict_label_list))
    elapsed_time = (time.time() - start) / 60 / 60
    if acc2 > 0.7:
        cmd = "cd {} && mv {} model_{}".format(os.path.join(path, "textcnn_model"), "base", acc2)
    else:
        cmd = "cd {} && rm -rf {}".format(os.path.join(path, "textcnn_model"), "base")
    logging.info("==========")
    logger.info(cmd)
    try:
        os.system(cmd)
    except:
        pass
    logger.info("The total program takes {} hours =and top2 acc is {}".format(elapsed_time,acc2))
    hyper["top1"]=acc1
    hyper["cmd"]=cmd
    hyper["top2"]=acc2
    with open(os.path.join(path, "textcnn_model","all_history"),"a",encoding="utf8") as f:
        f.writelines(json.dumps(hyper)+'\n')
    return acc2

def main_class():
    start = time.time()
    # Loads parameters from json file
    vocab_dict = id_word_map()
    with open(FLAGS.params_file) as f:
        config = json.load(f)
    config["train_size"]=26320
    config["max_length"] = 250
    config["id_word"] = vocab_dict
    config["word_dim"] = 300
    if config["train_size"] < FLAGS.shuffle_buffer_size:
        FLAGS.shuffle_buffer_size = config["train_size"]

    train_steps = int(config["train_size"] / FLAGS.batch_size * FLAGS.num_epoches)
    logger.info('The number of training steps is {}'.format(train_steps))
    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=FLAGS.steps_check, session_config=session_config,
                                        keep_checkpoint_max=3,tf_random_seed=1)
    num_warmup_steps = int(train_steps * FLAGS.warmup_proportion)
    early_stop_steps = int(train_steps * 0.4)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            "word_dim": config["word_dim"],
            "id_word": config["id_word"],
            "train_size": config["train_size"],
            'max_length': config["max_length"],
            'emb_file': FLAGS.emb_file,
            'learning_rate': FLAGS.learning_rate,
            'l2_reg_lambda': FLAGS.l2_reg_lambda,
            'dropout_prob': FLAGS.dropout_prob,
            'vocab': FLAGS.word_path,
            'num_filters': FLAGS.num_filters,
            'filter_sizes': list(map(int, FLAGS.filter_sizes.split(","))),
            'num_warmup_steps': num_warmup_steps,
            'train_steps': train_steps,
            'summary_dir': model_dir,
            "label_size": 125,
            "activation": "gelu",
            'use_focal_loss': False,
            'use_author_feature': False,
            'use_category_feature': False,
            'use_keyword_feature': False,
            'feature_dim': FLAGS.feature_dim
        }
    )
    no_increase_steps = int(config["train_size"] / FLAGS.batch_size * FLAGS.early_stop_epoches)

    # 用于early stop
    early_stop_hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, metric_name='f1',
                                                                    max_steps_without_increase=no_increase_steps,
                                                                   min_steps=early_stop_steps, run_every_secs=120)
    acc2=0

    # timeline_hook = tf.train.ProfilerHook(save_steps=FLAGS.steps_check, output_dir=model_dir + '/timeline/')
    if FLAGS.do_train == True:
        input_fn_for_train = lambda: input_fn(FLAGS.train_file, config, FLAGS.shuffle_buffer_size)
        train_spec = tf.estimator.TrainSpec(input_fn=input_fn_for_train, max_steps=train_steps)
        input_fn_for_eval = lambda: input_fn(FLAGS.valid_file, config, 0)
        # best_copier = BestCheckpointCopier(name='best',  # directory within model directory to copy checkpoints to
        #         checkpoints_to_keep=1,  # number of checkpoints to keep
        #         score_metric='acc',  # metric to use to determine "best"
        #         compare_fn=lambda x, y: x.score > y.score,
        #         sort_reverse=True)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=1200)  # exporters=best_copier
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        logger.info("Switch to the current directory and Run the command line:" \
                    "tensorboard --logdir=%s" \
                    "\nThen open http://localhost:6006/ into your web browser" % timestamp)
        logger.info("after train and evaluate")
    if FLAGS.do_predict == True:
        best_dir = model_dir + '/best'

        path_label = os.path.join(FLAGS.data_dir, 'textcnn_label_sort')
        with open(path_label, 'r', encoding='utf8') as f:
            lines = f.readlines()
            id2label = {i: l.strip().split("\x01\t")[0] for i, l in enumerate(lines)}

        # predict
        predict_label_list = []
        true_label_list = []
        prob_list=[]
        true_label_code=[]
        input_fn_for_test = lambda: input_fn(FLAGS.valid_file, config, 0)
        output_results = estimator.predict(input_fn_for_test, checkpoint_path=tf.train.latest_checkpoint(best_dir))

        with open(FLAGS.result_file, 'w') as writer:
            for prediction in output_results:
                predict_label_id = prediction["predict_label_ids"]
                true_label_id = prediction["true_label_ids"]
                prob_list.append(prediction["probabilities"])
                predict_label = id2label[predict_label_id]
                true_label = id2label[true_label_id]
                predict_label_list.append(predict_label)
                true_label_code.append(true_label_id)
                true_label_list.append(true_label)
                writer.write(predict_label + '\t' + true_label + "\n")
        test_y_name, test_preds_code=top_2_label_code(prob_list, true_label_code)
        acc2=accuracy_score( test_y_name, test_preds_code)
        logger.info(best_dir)
        logger.info("The total program acc1 {} and top2 acc is {}".format(accuracy_score(true_label_list, predict_label_list), acc2))

        # logger.info(classification_report(true_label_list, predict_label_list))
    elapsed_time = (time.time() - start) / 60 / 60
    if acc2 > 0.7:
        cmd = "cd {} && mv {} model_{}".format(os.path.join(path, "textcnn_model"), "base", acc2)
    else:
        cmd = "cd {} && rm -rf {}".format(os.path.join(path, "textcnn_model"), "base")
    logging.info("==========")
    logger.info(cmd)
    os.system(cmd)
    logger.info("The total program takes {} hours =and top2 acc is {}".format(elapsed_time,acc2))



    return acc2


if __name__ == '__main__':
    main_class()