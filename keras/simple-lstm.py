import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
import os
import sys
import pickle
import logging

currentUrl = os.path.dirname(__file__)
most_parenturl = os.path.abspath(os.path.join(currentUrl, os.pardir))
m_p, m_c = os.path.split(most_parenturl)
while 'xunfei' not in m_c:
    m_p, m_c = os.path.split(m_p)
sys.path.append(os.path.join(m_p, m_c))
from class_model.load_data import train_path, test_path, pred_path

EMBEDDING_FILES = [
   # '/data/tanggp/xun_class/aichallenge/tencent',
"/data/tanggp/xun_class/cc.zh.300.bin"
]
NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220
# 1
TEXT_COLUMN = 'jieba'
TARGET_COLUMN = 'label'
CHARS_TO_REMOVE = ''''!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'''

embedding_size=300
def build_matrix(word_index, emb_path):
    embedding_table = np.zeros((len(word_index) + 1, embedding_size))
    if '.bin' in emb_path:
        import fasttext as ft
        word2vec_model = ft.load_model(emb_path)
    else:
        from gensim.models import KeyedVectors
        word2vec_model = KeyedVectors.load_word2vec_format(emb_path, binary=False, limit=100)
        # word2vec_model = KeyedVectors.load(path, mmap='r')
    # n_words = len(word_index)
    # embedding_table = np.random.normal(loc=0.0, scale=initializer_range, size=(n_words, embedding_size))
    # print('Loading pretrained embeddings from {}...'.format(emb_path))

    emb_invalid = 0
    for word, i in word_index.items():
        embedding = None
        if '.bin' in emb_path:
            embedding = word2vec_model.get_word_vector(word)
        else:
            for candidate in [word, word.lower()]:
                if candidate in word2vec_model:
                    embedding = word2vec_model[candidate]
                    break
        if embedding is not None:
            embedding_table[i] = embedding
        else:
            emb_invalid += 1
    print("emb_invalid{}".format(emb_invalid))
    return embedding_table


def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    aux_result = Dense(num_aux_targets, activation='softmax')(hidden)

    model = Model(inputs=words, outputs=aux_result)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


train_df = pd.read_json(train_path, lines=True, encoding='utf8')
test_df = pd.read_json(test_path, lines=True, encoding='utf8')
pred_df = pd.read_json(pred_path, lines=True, encoding='utf8')

x_train = train_df[TEXT_COLUMN].astype(str)
x_test = test_df[TEXT_COLUMN].astype(str)
x_pred = pred_df[TEXT_COLUMN].astype(str)

# y_train = train_df[TARGET_COLUMN]
# y_test = test_df[TARGET_COLUMN]
label = train_df.groupby(TARGET_COLUMN).count()
label_dict = {la: i for i, la in enumerate(label.index.tolist())}
num_label = len(label_dict)

y_train = train_df[TARGET_COLUMN].map(label_dict)
y_test = train_df[TARGET_COLUMN].map(label_dict)
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False)
tokenizer.fit_on_texts(list(x_train) + list(x_test) + list(x_pred))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_pred = tokenizer.texts_to_sequences(x_pred)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
x_pred = sequence.pad_sequences(x_pred, maxlen=MAX_LEN)

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

checkpoint_predictions = []
weights = []

for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, num_label)
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=256))
        weights.append(2 ** global_epoch)

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)
