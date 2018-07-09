import collections
import os
import string

import numpy as np
import pandas as pd
from keras import losses
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, GRU
from keras.layers import Input, Dense
from keras.layers.embeddings import Embedding
# import keras
from keras.models import Model
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords
# import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# import Attention
from cdc.src.Attention import *


def normalize_text(texts):
    texts = [x.lower() for x in texts]
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    texts = [' '.join([word for word in x.split() if word not in set(stopwords.words("english"))]) for x in texts]
    texts = [' '.join(x.split()) for x in texts]
    return (texts)


def build_dictionary(sentences, vocabulary_size):
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    count = [['RARE', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return (word_dict)


def text_to_numbers(sentences, word_dict):
    data = []
    for sentence in sentences:
        sentence_data = []
        for word in sentence.split():
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return (data)


def build_embedding(embedding_file):
    embeddings_index = {}
    f = open(embedding_file, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_dictionary) + 1, embedding_size))
    for i, word in word_dictionary_rev.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(top_words, embedding_size, input_length=max_words)
    embedding_layer = Embedding(len(word_dictionary) + 1,
                                embedding_size,
                                weights=[embedding_matrix],
                                input_length=max_words,
                                trainable=False)
    return embedding_layer


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch, 0])
    return activations


def multiclass_roc_auc_score(y_true, y_pred, average='macro'):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average="weighted")


def train(f_path, l_path, wemb_file):
    print('start')
    global vocabulary_size
    global max_words
    global top_words
    global embedding_size
    global word_dictionary
    global word_dictionary_rev

    vocabulary_size = 1000000
    max_words = 300
    top_words = 1000000
    embedding_size = 300
    attention = True

    print("--- Loading data ---")
    texts = pd.read_csv(f_path, sep='\t', encoding='latin-1')
    texts = texts.ix[:, 1].values.tolist()
    label = pd.read_csv(l_path, sep='\t', header=None, encoding='latin-1')
    label = label.values.tolist()
    target = label
    print("Read %d rows of data" % len(texts))
    print("Read %d rows of label" % len(label))

    # label encoder
    print("--- Label encoding ---")
    encoder = LabelEncoder()
    encoder.fit(target)
    target = encoder.transform(target)
    target = to_categorical(target)

    # text preprocessing
    print("--- Text processing ---")
    texts = normalize_text(texts)
    word_dictionary = build_dictionary(texts, vocabulary_size)
    word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))

    text_data = text_to_numbers(texts, word_dictionary)
    train_indices = np.random.choice(len(text_data), int(round(0.7 * len(text_data))), replace=False)
    test_indices = np.array(list(set(range(len(text_data))) - set(train_indices)))
    texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
    texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
    target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
    target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
    # Convert texts to lists of indices
    text_data_train = np.array(text_to_numbers(texts_train, word_dictionary))
    text_data_test = np.array(text_to_numbers(texts_test, word_dictionary))
    # Pad/crop
    text_data_train = np.array([x[0:max_words] for x in [y + [0] * max_words for y in text_data_train]])
    text_data_test = np.array([x[0:max_words] for x in [y + [0] * max_words for y in text_data_test]])

    # embedding
    # https://fasttext.cc/docs/en/english-vectors.html
    embedding_layer = build_embedding(wemb_file)

    # model construction
    print("--- Build model ---")
    sequence_input = Input(shape=(max_words,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    if attention == False:
        m = Bidirectional(GRU(128))(embedded_sequences)
    else:
        m = Bidirectional(GRU(128, return_sequences=True, consume_less='mem'))(embedded_sequences)
        m = Attention()(m)
    pred = Dense(len(encoder.classes_), activation='softmax')(m)
    model = Model(sequence_input, pred)
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(),
                  metrics=['accuracy', precision, recall, fmeasure])
    print(model.summary())

    # training
    print("--- Training ---")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_fit = model.fit(text_data_train, target_train,
                          validation_data=(text_data_test, target_test),
                          epochs=100,
                          batch_size=64,
                          shuffle=True,
                          callbacks=[early_stopping])
    history = pd.DataFrame(model_fit.history)
    history.to_csv("model_history.txt", sep="\t")

    # prediction
    print("--- Prediction ---")
    scores = model.evaluate(text_data_test, target_test, verbose=0)
    y_pred = model.predict(text_data_test)
    auc = roc_auc_score(target_test, y_pred, average='weighted')
    print(scores[1], scores[2], scores[3], scores[4], auc)

    # save model
    print("--- Save model ---")
    json_model = model.to_json()
    open('model.json', 'w').write(json_model)
    model.save_weights('model_weights.h5', overwrite=True)
    # cPickle.dump(encoder, open('model_encoder.pkl', 'wb'))

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend