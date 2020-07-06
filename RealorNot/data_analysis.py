# anaconda/learn python
# -*- coding: utf-8 -*-

"""
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'data_analysis.py'
@Time: '2020/6/6 17:01'
@Desc : 'kaggle，根据twitter内容预测是否为真实的风险信息，数据分析'
"""

import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))


def create_corpus(dataset, target):
    """
    @Desc : '创建语料库'
    @Parameters :
        'dataset' - '读取到的文件数据'
        'target' - '两种分类标签，如0和1'
    @Returns :
        'corpus' - '所有单词的list'
    @Time : '2020/6/7 16:54'
    """
    corpus = []

    for x in dataset[dataset['target'] == target]['text'].str.split():
        for i in x:
            corpus.append(i)

    return corpus


def get_top_tweet_bigram(corpus, n=None):
    """
    @Desc : '进行bigram分析'
    @Parameters :
        'corpus' - '语料库'
    @Returns :
        'words_freq' - '常出现的bigram词组'
    @Time : '2020/6/7 17:27'
    """
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n]


if __name__ == '__main__':
    # 设置pandas显示全部列
    pd.set_option('display.max_columns', None)
    # 设置显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    origins_train_dataset = pd.read_csv('./dataset/train.csv')

    # 使用nltk创建停用词list
    stop_words = set(stopwords.words('english'))

    # 绘制柱状图，查看target分布
    target_counts = origins_train_dataset['target'].value_counts()
    sns.barplot(target_counts.index, target_counts)
    plt.xticks(target_counts.index)
    plt.title("数据类别分布")
    for i in target_counts.index:
        plt.text(i, target_counts[i] + 0.05, '%d' % target_counts[i], ha='center', va='bottom', fontsize=11)
    plt.show()

    # 字母级的统计分析，查看大部分推特内容的长度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # 统计两种分类情况下的，每条推特内容的字母长度
    tweet_len_1 = origins_train_dataset[origins_train_dataset['target'] == 1]['text'].str.len()
    tweet_len_0 = origins_train_dataset[origins_train_dataset['target'] == 0]['text'].str.len()

    # 横坐标为字母数，纵坐标为推特条数
    ax1.hist(tweet_len_1, color='red')
    ax2.hist(tweet_len_0, color='green')
    ax1.set_title('真实的推文（1）')
    ax2.set_title('不真实的推文（0）')
    fig.suptitle('推文字母长度')
    plt.show()

    # 单词级统计分析，查看大部分推特的单词数
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # 统计每条推特的单词个数x
    tweet_word_len_1 = origins_train_dataset[origins_train_dataset['target'] == 1]['text'].str.split().map(
        lambda x: len(x))
    tweet_word_len_0 = origins_train_dataset[origins_train_dataset['target'] == 0]['text'].str.split().map(
        lambda x: len(x))

    # 横坐标为单词个数，纵坐标为推特条数
    ax1.hist(tweet_word_len_1, color='red')
    ax2.hist(tweet_word_len_0, color='green')
    ax1.set_title('真实的推文（1）')
    ax2.set_title('不真实的推文（0）')
    fig.suptitle('推文单词长度')
    plt.show()

    # 每条推特的平均单词长度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    tweet_average_word_len_1 = origins_train_dataset[origins_train_dataset['target'] == 1]['text'].str.split().apply(
        lambda x: [len(i) for i in x])
    tweet_average_word_len_0 = origins_train_dataset[origins_train_dataset['target'] == 0]['text'].str.split().apply(
        lambda x: [len(i) for i in x])
    sns.distplot(tweet_average_word_len_1.map(lambda x: np.mean(x)), ax=ax1, color='red')
    sns.distplot(tweet_average_word_len_0.map(lambda x: np.mean(x)), ax=ax2, color='green')
    ax1.set_title('真实的推文（1）')
    ax2.set_title('不真实的推文（0）')
    fig.suptitle('推文的平均单词长度')
    plt.show()

    # 创建tweet的停用词字典
    tweet_corpus_0 = create_corpus(origins_train_dataset, 0)
    tweet_corpus_1 = create_corpus(origins_train_dataset, 1)

    tweet_corpus_dict_0 = defaultdict(int)
    tweet_corpus_dict_1 = defaultdict(int)
    for word in tweet_corpus_0:
        if word in stop_words:
            tweet_corpus_dict_0[word] += 1
    for word in tweet_corpus_1:
        if word in stop_words:
            tweet_corpus_dict_1[word] += 1

    # 将停用词按出现次数排序
    tweet_corpus_dict_0 = sorted(tweet_corpus_dict_0.items(), key=lambda x: x[1], reverse=True)[:10]
    tweet_corpus_dict_1 = sorted(tweet_corpus_dict_1.items(), key=lambda x: x[1], reverse=True)[:10]

    # 停用词频率图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    tweet_corpus_dict_0_x, tweet_corpus_dict_0_y = zip(*tweet_corpus_dict_0)
    tweet_corpus_dict_1_x, tweet_corpus_dict_1_y = zip(*tweet_corpus_dict_1)
    sns.barplot(list(tweet_corpus_dict_0_x), tweet_corpus_dict_0_y, ax=ax1)
    sns.barplot(list(tweet_corpus_dict_1_x), tweet_corpus_dict_1_y, ax=ax2)
    ax1.set_title('真实的推文（1）')
    ax2.set_title('不真实的推文（0）')
    fig.suptitle('推文的停用词频率')
    plt.show()

    # 分析标点符号
    tweet_corpus_punct_dic_0 = defaultdict(int)
    tweet_corpus_punct_dic_1 = defaultdict(int)

    special_punctuation = string.punctuation
    for i in tweet_corpus_0:
        if i in special_punctuation:
            tweet_corpus_punct_dic_0[i] += 1
    for i in tweet_corpus_1:
        if i in special_punctuation:
            tweet_corpus_punct_dic_1[i] += 1

    tweet_corpus_punct_dic_0 = sorted(tweet_corpus_punct_dic_0.items(), key=lambda x: x[1], reverse=True)[:15]
    tweet_corpus_punct_dic_1 = sorted(tweet_corpus_punct_dic_1.items(), key=lambda x: x[1], reverse=True)[:15]

    # 标点符号频率图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    tweet_corpus_punct_dic_0_x, tweet_corpus_punct_dic_0_y = zip(*tweet_corpus_punct_dic_0)
    tweet_corpus_punct_dic_1_x, tweet_corpus_punct_dic_1_y = zip(*tweet_corpus_punct_dic_1)
    sns.barplot(list(tweet_corpus_punct_dic_0_x), tweet_corpus_punct_dic_0_y, ax=ax1)
    sns.barplot(list(tweet_corpus_punct_dic_1_x), tweet_corpus_punct_dic_1_y, ax=ax2)
    ax1.set_title('真实的推文（1）')
    ax2.set_title('不真实的推文（0）')
    fig.suptitle('推文的标点符号频率')
    plt.show()

    # 统计推文中的单词重复
    counter_0 = Counter(tweet_corpus_0)
    counter_1 = Counter(tweet_corpus_1)
    most_0 = counter_0.most_common()
    most_1 = counter_1.most_common()
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    for word, count in most_0[:50]:
        if word not in stop:
            x_0.append(word)
            y_0.append(count)
    for word, count in most_1[:40]:
        if word not in stop:
            x_1.append(word)
            y_1.append(count)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=x_0, y=y_0, ax=ax1)
    sns.barplot(x=x_1, y=y_1, ax=ax2)
    ax1.set_title('不真实的推文（0）')
    ax2.set_title('真实的推文（1）')
    fig.suptitle('推文中单词重复率')
    plt.show()

    # 查看内容的bigram分析
    plt.figure(figsize=(10, 5))
    top_tweet_bigram = get_top_tweet_bigram(origins_train_dataset['text'])[:20]
    x, y = map(list, zip(*top_tweet_bigram))
    sns.barplot(x=y, y=x)
    plt.suptitle('推文bigram分析')
    plt.show()
