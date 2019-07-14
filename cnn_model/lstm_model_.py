#!/usr/bin/python3
# -*- coding: utf-8 -*-
from cnn_model.model import Model
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell


import tensorflow as tf

def gelu(input_tensor):
     """Gaussian Error Linear Unit.
     This is a smoother version of the RELU.
     Original paper: https://arxiv.org/abs/1606.08415
     Args:
        input_tensor: float Tensor to perform activation.
     Returns:
       `input_tensor` with the GELU activation applied.
     """
     cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
     return input_tensor * cdf


class LSTMModel(Model):
    def __init__(self, config, input_ids, author_id,  is_training):
        super(LSTMModel, self).__init__(config, input_ids, is_training)
        self.author_id = author_id


    def build_network(self):
        """
        Build network function.
        """
        embedding = self.embedding_lookup(self.input_ids, self.config['id_word'], embedding_size=self.config['word_dim'], initializer_range=0.02,
                         word_embedding_name="embedding_table", emb_path=self.config['emb_file'])
        self.variable_summaries('embedding', embedding)
        embedded_words_expanded = self. expand_dims(embedding, -1)
        logits, predict_label_ids, l2_loss,probabilities = self.build_lstm(embedded_words_expanded)
        return logits, predict_label_ids, l2_loss,probabilities
    #build_loss(self, labels, logits, l2_loss=0.0)
    def build_loss(self, labels, logits, l2_loss=0):
        """Build loss function.
        args:
          labels: Actual label.
          logits:
        """
        # Loss
        with tf.variable_scope("loss"):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                loss = tf.reduce_mean(losses) + self.config['l2_reg_lambda']*l2_loss
        return loss

    def build_lstm(self,input_tensor):

        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.config["lstm_para"]["hidden_size"]),
                                BasicLSTMCell(self.config["lstm_para"]["hidden_size"]),
                                inputs=input_tensor, dtype=tf.float32)
        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.config["lstm_para"]["hidden_size"]], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.config["lstm_para"]["hidden_size"]]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.config["lstm_para"]["max_len"])))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(alpha, [-1, self.config["lstm_para"]["max_len"], 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.config["lstm_para"]["keep_prob"])

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.config["lstm_para"]["hidden_size"], self.config['label_size']], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.config['label_size']]))
        logits = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)


        # prediction
        probabilities = tf.argmax(tf.nn.softmax(logits), 1)
        predict_label_ids = tf.argmax(logits, axis=1, name="predict_label_id")  # 预测结果

        pooled_outputs = []
        l2_loss = tf.constant(0.0)  # 先不用，写0
        l2_loss += tf.nn.l2_loss(FC_W) + tf.nn.l2_loss(FC_b)
        # with tf.variable_scope("output"):
        #     output_w = tf.get_variable("output_w", shape=[hidden_size, self.config['label_size']])
        #     output_b =  self.initialize_bias("output_b", shape=self.config['label_size'])
        #     logits = tf.nn.xw_plus_b(output_layer, output_w, output_b)
        #
        # probabilities = tf.nn.softmax(logits, axis=-1)
        # predict_label_ids = tf.argmax(logits, axis=1, name="predict_label_id")  # 预测结果
        return logits, predict_label_ids, l2_loss,probabilities

