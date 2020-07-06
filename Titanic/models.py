# anaconda/learn python
# -*- coding: utf-8 -*-

"""
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'data_analysis.py'
@Time: '2020/6/5 19:53'
@Desc : 'kaggle新手项目，泰坦尼克号幸存者预测。建模文件'
"""

import pandas as pd
import tensorflow.keras as tfk

if __name__ == '__main__':
    # 显示所有列
    pd.set_option('display.max_columns', None)
    origins_dataset = pd.read_csv("./dataset/train.csv")

    y_train = origins_dataset['Survived']
    x_train = origins_dataset.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'],
                                   axis=1)

    sex_list = x_train['Sex'].unique()
    embarked_list = x_train['Embarked'].unique()

    # 将字符串变量转变为数值标签
    x_train.loc[x_train['Sex'] == 'male', 'Sex'] = 0
    x_train.loc[x_train['Sex'] == 'female', 'Sex'] = 1

    x_train.loc[x_train['Embarked'] == 'S', 'Embarked'] = 0
    x_train.loc[x_train['Embarked'] == 'C', 'Embarked'] = 1
    x_train.loc[x_train['Embarked'] == 'Q', 'Embarked'] = 2

    # 填充缺失值
    x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())
    x_train['Embarked'] = x_train['Embarked'].mode()[0]

    x_train['Sex'] = x_train['Sex'].astype('int64')
    x_train['Embarked'] = x_train['Embarked'].astype('int64')

    # 查看描述统计，主要查看是否有缺失情况
    # print(x_train.describe(include='all'))

    # 建立深度神经网络模型。该模型准确率只有75%左右。
    model = tfk.Sequential([
        tfk.layers.Dense(100, activation='relu'),
        tfk.layers.Dense(50, activation='relu'),
        tfk.layers.Dense(10, activation='relu'),
        tfk.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=tfk.optimizers.RMSprop(lr=0.001), metrics=['acc'])

    x_train = x_train.values
    y_train = y_train.values

    model.fit(x_train, y_train, validation_split=0.33, epochs=100)

    model.save('./titanic.h5')
