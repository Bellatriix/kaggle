# anaconda/learn python
# -*- coding: utf-8 -*-

"""
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'preprocessing.py'
@Time: '2020/6/8 8:01'
@Desc : 'kaggle，根据twitter内容预测是否为真实的风险信息，预处理'
"""

import numpy as np
import pandas as pd
import tensorflow.keras as tfk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))


def create_corpus(df):
    """
    @Desc : '根据输入的dataframe的文本内容，创建语料库'
    @Parameters :
        'df' - '读取csv文件后，并清洗完成的dataframe'
    @Returns :
        'corpus' - '建立的语料库'
    @Time : '2020/6/8 9:06'
    """
    # list预留空间时，更新会更快，如果新建空list，再append，会特别慢。时间应该消耗在append上
    # corpus = []
    # words = [0] * 1000
    # k = 0
    # tqdm是进度条包
    corpus = []

    # for tweet in tqdm(df['text']):
    #     words = [word.lower() for word in word_tokenize(tweet) if word.isalpha() == 1 and word not in stop]
    #     corpus.append(words)

    # 此处words应放在循环内，每次循环记录临时的句子，如果放在外部则不会清空上一次的句子
    for tweet in tqdm(df['text']):
        words = []
        for word in word_tokenize(tweet):
            if (word.isalpha() == 1) and (word not in stop):
                words.append(word.lower())

        corpus.append(words)

    return corpus


if __name__ == '__main__':
    training_set = pd.read_csv('./dataset/train_cleaning.csv')
    test_set = pd.read_csv('./dataset/test_cleaning.csv')

    # 停用词过滤
    print("stop words")
    all_dataset = pd.concat([training_set, test_set])
    corpus_dataset = create_corpus(all_dataset)

    # 嵌入
    print("embedding")
    embedding_dict = {}
    with open('./dataset/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], 'float32')
            embedding_dict[word] = vectors
    f.close()

    # 分词
    print("segmentation")
    MAX_LEN = 50
    tokenizer_obj = tfk.preprocessing.text.Tokenizer()
    tokenizer_obj.fit_on_texts(corpus_dataset)
    sequences = tokenizer_obj.texts_to_sequences(corpus_dataset)

    # 转变序列
    tweet_pad = tfk.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
    word_index = tokenizer_obj.word_index

    # print(len(word_index))

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, 100))

    # word_index  dict
    word_index_save = pd.DataFrame(word_index, index=[0])
    word_index_save.to_csv('./dataset/word_index.csv', index=False)

    # 生成词向量
    print("word vector")
    for word, i in tqdm(word_index.items()):
        if i > num_words:
            continue

        emb_vec = embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec

    embedding_matrix_save = np.save('./dataset/embedding_matrix.npy', embedding_matrix)
    embedding_matrix_file = np.load('./dataset/embedding_matrix.npy')

    # 判断保存的embedding矩阵是否一致
    embedding_diff = embedding_matrix - embedding_matrix_file
    print(np.argwhere(embedding_diff > 0))

    # 分开训练集和测试集，ndarry
    train_set = tweet_pad[:training_set.shape[0]]
    test_set = tweet_pad[training_set.shape[0]:]

    np.save('./dataset/train_set.npy', train_set)
    np.save('./dataset/test_set.npy', test_set)

    train_set_file = np.load('./dataset/train_set.npy')
    test_set_file = np.load('./dataset/test_set.npy')

    # 判断保存的数据集是否一致
    if (train_set == train_set_file).all():
        print("train set true")
    else:
        print("train set false")

    if (test_set == test_set_file).all():
        print("test set true")
    else:
        print("test set false")
