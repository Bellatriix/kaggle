# anaconda/learn python
# -*- coding: utf-8 -*-

"""
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'modeling.py'
@Time: '2020/6/8 19:36'
@Desc : 'kaggle，根据twitter内容预测是否为真实的风险信息，建模'
"""

import os
import numpy as np
import pandas as pd
import tensorboard
import tensorflow.keras as tfk
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    training_set = pd.read_csv('./dataset/train_cleaning.csv')
    train_set = np.load('./dataset/train_set.npy')

    word_index_df = pd.read_csv('./dataset/word_index.csv').T
    word_index = word_index_df.to_dict()
    word_index = word_index[0]
    # print(len(word_index))

    embedding_matrix = np.load('./dataset/embedding_matrix.npy')
    # print(embedding_matrix.shape)

    MAX_LEN = 50
    num_words = len(word_index) + 1

    print("start modeling")
    # 建立模型
    model = tfk.Sequential()

    print("layers")
    embedding = tfk.layers.Embedding(num_words, 100,
                                     embeddings_initializer=tfk.initializers.Constant(embedding_matrix),
                                     input_length=MAX_LEN,
                                     trainable=False)
    """
    # 基本模型
    model.add(embedding)
    model.add(tfk.layers.SpatialDropout1D(0.2))
    model.add(tfk.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(tfk.layers.Dense(1, activation='sigmoid'))
    """

    # 加入双向传播，卷积和池化
    model.add(embedding)
    model.add(tfk.layers.Bidirectional(tfk.layers.LSTM(100, return_sequences=True)))
    model.add(tfk.layers.SpatialDropout1D(0.2))
    model.add(tfk.layers.Conv1D(64, 5, activation='relu'))
    model.add(tfk.layers.MaxPooling1D(pool_size=5))
    model.add(tfk.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(tfk.layers.Dense(10, activation='relu', kernel_regularizer=tfk.regularizers.l2(0.01)))
    model.add(tfk.layers.Dense(1, activation='sigmoid'))

    optimizer = tfk.optimizers.Adam(learning_rate=1e-5)

    print("compile")
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

    # 查看模型结构
    # model.summary()
    # exit()

    # 从训练集中划分交叉验证集
    X_train, X_test, Y_train, Y_test = train_test_split(train_set, training_set['target'].values,
                                                        test_size=0.15)

    print("fit")
    # 训练之前，记录模型数据
    # 此处报错可能与windows和tf的内部实现有关，必须要加os.path.join，并且要用反斜杠。
    logdir = os.path.join("logs\\conv_and_pool")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=logdir)

    history = model.fit(X_train, Y_train,
                        batch_size=4,
                        epochs=15,
                        validation_data=(X_test, Y_test),
                        verbose=2,
                        callbacks=[tensorboard_callback])

    model.save('./models/RealorNot_convandpool.h5')

    print("end of model")
