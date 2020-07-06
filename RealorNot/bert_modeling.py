# anaconda/learn python
# -*- coding: utf-8 -*-

"""
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'bert_modeling.py'
@Time: '2020/6/20 14:50'
@Desc : ''
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_hub as hub
from RealorNot import tokenization


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):
    input_word_ids = tfk.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tfk.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tfk.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = tfk.layers.Dense(1, activation='sigmoid')(clf_output)

    model = tfk.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tfk.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metric=['acc'])

    return model


if __name__ == '__main__':
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2"
    bert_layer = hub.KerasLayer(module_url, trainable=True)

    print("read data")
    train = pd.read_csv("./dataset/train.csv")
    test = pd.read_csv("./dataset/test.csv")

    vocab_file = bert_layer.resolved_object.vocab_file.assert_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    print("pre data")
    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    test_input = bert_encode(test.text.values, tokenizer, max_len=160)
    train_labels = train.target.values

    model = build_model(bert_layer, max_len=160)
    # model.summary()

    print("start modeling")
    logdir = os.path.join("logs\\bert")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=logdir)

    history = model.fit(train_input, train_labels,
                        validation_split=0.2,
                        epochs=5,
                        callbacks=[tensorboard_callback],
                        batch_size=16)

    model.save('./models/RealorNot_bert.h5')

    print("end of model")
