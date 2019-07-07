"""Export model as a saved_model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
import json

import tensorflow as tf

from run_classify import *
import os
# DATADIR = '../../data/example'
# PARAMS = './results/params.json'
MODELDIR = '/data/tanggp/nsfw_out2'

# try:
#     os.makedirs(MODELDIR)
# except:
#     pass
def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    # words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    # nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    # # receiver_tensors = {'words': words, 'nwords': nwords}
    # features = {'words': words, 'nword1s': nwords}

    seq_length=FLAGS.max_seq_length
    name_to_features_ = {
        "input_ids": tf.placeholder(dtype=tf.int64, shape=[None, seq_length]) ,# tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.placeholder(dtype=tf.int64, shape=[None, seq_length]),
        "segment_ids": tf.placeholder(dtype=tf.int64, shape=[None, seq_length]),
        "label_ids": tf.placeholder(dtype=tf.int64, shape=[None]),
        "image": tf.placeholder(dtype=tf.float32, shape=[None, 2048]),
    }
    return tf.estimator.export.ServingInputReceiver(name_to_features_, name_to_features_)


if __name__ == '__main__':
    # with Path(PARAMS).open() as f:
    #     params = json.load(f)

    # params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    # params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    # params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    # params['glove'] = str1(Path(DATADIR, 'glove.npz'))
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    num_train_steps = None
    num_warmup_steps = None
    run_config =tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    model_fn=model_fn_builder_gpu(
        bert_config=bert_config,
        num_labels=FLAGS.num_lables,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)


    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    estimator.export_saved_model(os.path.join(FLAGS.output_dir,'saved_model'), serving_input_receiver_fn)
