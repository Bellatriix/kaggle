# anaconda/learn python
# -*- coding: utf-8 -*-

"""
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'data_cleaning.py'
@Time: '2020/6/8 11:19'
@Desc : 'kaggle，根据twitter内容预测是否为真实的风险信息，数据清洗'
"""

import re
import string
import pandas as pd
from nltk.corpus import stopwords
from spellchecker import SpellChecker

stop = set(stopwords.words('english'))


def remove_url(text):
    """
    @Desc : '移除文本内容中的网络地址链接'
    @Parameters :
        'text' - '文本内容'
    @Returns :
        'text' - '处理后的文本内容'
    @Time : '2020/6/8 8:14'
    """
    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'', text)


def remove_html(text):
    """
    @Desc : '移除文本内容中的html标签'
    @Parameters :
        'text' - '文本内容'
    @Returns :
        'text' - '处理后的内容'
    @Time : '2020/6/8 8:19'
    """
    html = re.compile(r'<.*?>')

    return html.sub(r'', text)


def remove_emoji(text):
    """
    @Desc : '移除文本内容中的表情'
    @Parameters :
        'text' - '文本内容'
    @Returns :
        'text' - '处理后的内容'
    @Time : '2020/6/8 8:24'
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_punct(text):
    """
    @Desc : '移除文本中的标点符号'
    @Parameters :
        'text' - '文本内容'
    @Returns :
        'text' - '处理后的内容'
    @Time : '2020/6/8 8:30'
    """
    punct = str.maketrans('', '', string.punctuation)

    return text.translate(punct)


def correct_spellings(text):
    """
    @Desc : '检查文本内容的拼写错误'
    @Parameters :
        'text' - '文本内容'
    @Returns :
        'correct_text' - '处理后的文本'
    @Time : '2020/6/8 8:55'
    """
    spell = SpellChecker()
    correct_text = []
    misspelled_words = spell.unknown(text.split())

    for word in text.split():
        if word in misspelled_words:
            correct_text.append(spell.correction(word))
        else:
            correct_text.append(word)

    return " ".join(correct_text)


if __name__ == '__main__':
    origins_train_dataset = pd.read_csv('./dataset/train.csv')
    test_dataset = pd.read_csv('./dataset/test.csv')

    all_dataset = pd.concat([origins_train_dataset, test_dataset])

    all_dataset['text'] = all_dataset['text'].apply(lambda x: remove_url(x))
    all_dataset['text'] = all_dataset['text'].apply(lambda x: remove_html(x))
    all_dataset['text'] = all_dataset['text'].apply(lambda x: remove_emoji(x))
    all_dataset['text'] = all_dataset['text'].apply(lambda x: remove_punct(x))
    all_dataset['text'] = all_dataset['text'].apply(lambda x: correct_spellings(x))

    # 分开训练集和测试集
    train_set = all_dataset[:origins_train_dataset.shape[0]]
    test_set = all_dataset[origins_train_dataset.shape[0]:]

    # 保存测试集到新文件
    train_set.to_csv('./dataset/train_cleaning.csv', index=False)
    test_set.to_csv('./dataset/test_cleaning.csv', index=False)

    print('suc')
