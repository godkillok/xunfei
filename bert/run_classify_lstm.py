# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.01
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

currentUrl = os.path.dirname(__file__)
most_parenturl = os.path.abspath(os.path.join(currentUrl, os.pardir))
m_p, m_c = os.path.split(most_parenturl)
while 'xunfei' not in m_c:
    m_p, m_c = os.path.split(m_p)
import sys

sys.path.append(os.path.join(m_p, m_c))
import collections
import csv
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import logging
import pandas as pd
import pickle
import json
from sklearn.metrics import classification_report, accuracy_score
from cnn_model.post_pred import post_pred, post_eval
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
flags = tf.flags
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FLAGS = flags.FLAGS

bert_model = '/home/tom/chinese/'
data_path = '/home/tom/swagaf/data'
output = '/tmp/xnli_output/'
## Required parameters
flags.DEFINE_string(
    "data_dir", data_path,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_model, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'SWAG', "The name of the task to train.")

flags.DEFINE_string("vocab_file", os.path.join(bert_model, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", output,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "label_path", output,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "history_dir", output,
    "The output directory where the model checkpoints will be written.")
## Other parameters

flags.DEFINE_string(
    "init_checkpoint", os.path.join(bert_model, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_integer(
    "num_gpu_cores", 4,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_eval_pred", True, "Whether to run eval on the dev set.")
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")
flags.DEFINE_float("drop_rate", 0.9, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 16, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_integer(
    "num_lables", 14,
    "num of labels ")
flags.DEFINE_integer(
    "eval_steps", 100,
    "elavalue steps")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, shuffle=False, multi_choice=1):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features_ = {
        "guid": tf.FixedLenFeature([], tf.string),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64)
    }
    if multi_choice > 1:
        name_to_features = {"label_ids": tf.FixedLenFeature([], tf.int64)}
        for i in range(multi_choice):
            for (k, v) in name_to_features_.items():
                if k != 'label_ids':
                    name_to_features[k + '_{}'.format(i)] = v

    else:
        name_to_features = name_to_features_

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            if t.dtype == tf.float64:
                t = tf.to_float(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        files = tf.data.Dataset.list_files(input_file, shuffle=shuffle)
        d = tf.data.TFRecordDataset(files)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a 1simple classification task on the entire
    # segment.
    #    name_to_features_ = {
    #     "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    #     "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    #     "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    #     "label_ids": tf.FixedLenFeature([], tf.int64),
    # }
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.12
    embedding = model.get_sequence_output() #[batch_size, seq_length, embedding_size]
    max_seq_length = embedding.shape[1].value

    print('======********======shape{}'.format(embedding.shape))
    #from keras.layers import  concatenate
    rnn_outputs, _ = bi_rnn(tf.nn.rnn_cell.LSTMCell(128),
                            tf.nn.rnn_cell.LSTMCell(128),
                            inputs=embedding, dtype=tf.float32)
    output_rnn = tf.concat(rnn_outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
    # rnn_outputs, _ = bi_rnn(BasicLSTMCell(128),
    #                         BasicLSTMCell(128),
    #                         inputs=output_rnn, dtype=tf.float32)
    #
    # output_rnn = tf.concat(rnn_outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
    print('======********======shape{}'.format(output_rnn.shape))
    GlobalMaxPooling1D=tf.layers.max_pooling1d (output_rnn, pool_size=[ FLAGS.max_seq_length],
                   strides=[1], padding="VALID", name="GlobalMaxPooling1D")# [batch_size,hidden_size*2]

    GlobalAveragePooling1D = tf.layers.average_pooling1d(output_rnn, pool_size=[FLAGS.max_seq_length],
                                        strides=[  1], padding="VALID", name="GlobalAveragePooling1D")
    hidden = tf.concat([GlobalMaxPooling1D,GlobalAveragePooling1D],2)
    hidden=tf.squeeze(hidden)
    hidden_size=hidden.shape[-1].value
    logging.info("hidden {}".format(hidden.shape))
    output_layer=tf.layers.dense(hidden, hidden_size, activation=modeling.gelu,name='dense_layer')
    z = tf.add(output_layer, hidden)

    output_layer = tf.nn.dropout(z , keep_prob=FLAGS.drop_rate)

    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=FLAGS.drop_rate)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        guid = features["guid"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        # squeeze_label_ids = tf.squeeze(label_ids, axis=1)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Tra2inable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                # recall=tf.metrics.recall(label_ids, predictions)
                # f1_score=tf.contrib.metrics.f1_score(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)

                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    # "eval_recall":recall,
                    # "eval_f1_score": f1_score,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            predict_label_ids = tf.argmax(logits, axis=1, name="predict_label_id")
            predictions = {
                'probabilities': probabilities,
                "guid": guid,
                'true_label_ids': label_ids,
                'predict_label_ids': predict_label_ids
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence1 length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    session_config.gpu_options.allow_growth = True
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    num_train_example = None
    if FLAGS.do_train:
        train_meta = os.path.join(FLAGS.data_dir, "train.json")
        with open(train_meta, 'r') as f:
            d = json.load(f)
        num_train_example = d['num_train_example']
        num_train_steps = int(
            num_train_example / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=FLAGS.num_lables,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.data_dir, "train*.tfrecord")

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True, shuffle=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_file = os.path.join(FLAGS.data_dir, "eval*.tfrecord")
        eval_meta = os.path.join(FLAGS.data_dir, "eval.json")

        with open(eval_meta, 'r') as f:
            d = json.load(f)
            num_eval_examples = d['num_eval_examples']

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", num_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = FLAGS.eval_steps
        if eval_steps == 0:
            eval_steps = None
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.1
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.1
            eval_steps = int(num_eval_examples / FLAGS.eval_batch_size)
        eval_steps = int(num_eval_examples / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_eval_pred:

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_file = os.path.join(FLAGS.data_dir, "eval*.tfrecord")
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        output_results = estimator.predict(input_fn=predict_input_fn)
        model_dir = FLAGS.output_dir
        path_label = FLAGS.label_path
        history_dir = FLAGS.history_dir
        acc2, acc1 = post_eval(path_label,  output_results,model_dir, history_dir)

        logging.info("The total program takes =and top2 acc is {}".format(acc2))

        if acc2 > 0.7:
            predict_file = os.path.join(FLAGS.data_dir, "pred*.tfrecord")
            predict_input_fn = file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder)
            output_results = estimator.predict(predict_input_fn)
            path_label = FLAGS.label_path
            history_dir = FLAGS.history_dir
            post_pred(path_label, model_dir, history_dir, output_results, acc2)

    if FLAGS.do_predict:
        pred_meta = os.path.join(FLAGS.data_dir, "predict.json")
        predict_file = os.path.join(FLAGS.data_dir, "predict*.tfrecord")
        with open(pred_meta, 'r') as f:
            d = json.load(f)
            num_pred_examples = d['num_pred_examples']

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", num_pred_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
