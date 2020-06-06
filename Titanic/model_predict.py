# anaconda/learn python
# -*- coding: utf-8 -*-

"""
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'model_predict.py'
@Time: '2020/6/6 13:58'
@Desc : 'kaggle新手项目，泰坦尼克号幸存者预测。预测文件'
"""

import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    test_origins_dataset = pd.read_csv('./dataset/test.csv')

    x_test = test_origins_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
                                       axis=1)

    # 填充缺失值
    x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())
    x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].mean())

    x_test.loc[x_test['Sex'] == 'male', 'Sex'] = 0
    x_test.loc[x_test['Sex'] == 'female', 'Sex'] = 1
    x_test['Sex'] = x_test['Sex'].astype('int64')

    x_test.loc[x_test['Embarked'] == 'S', 'Embarked'] = 0
    x_test.loc[x_test['Embarked'] == 'C', 'Embarked'] = 1
    x_test.loc[x_test['Embarked'] == 'Q', 'Embarked'] = 2
    x_test['Embarked'] = x_test['Embarked'].astype('int64')

    # print(x_test.describe(include='all'))

    x_test = x_test.values

    model = tfk.models.load_model('./titanic.h5')
    result_predict = model.predict(x_test)

    passenger_Id = test_origins_dataset['PassengerId'].to_frame()
    result_dataframe = pd.DataFrame(result_predict, columns=['Survived'])

    # 将预测的小数变为0或1
    result_dataframe.loc[result_dataframe['Survived'] < 0.5, 'Survived'] = 0
    result_dataframe.loc[result_dataframe['Survived'] >= 0.5, 'Survived'] = 1
    result_dataframe['Survived'] = result_dataframe['Survived'].astype('int64')

    result = pd.concat([passenger_Id, result_dataframe], axis=1)
    result.to_csv('./submission.csv', index=False)

    print('suc')
