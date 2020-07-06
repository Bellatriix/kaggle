# anaconda/learn python
# -*- coding: utf-8 -*-

"""
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'predict.py'
@Time: '2020/6/8 15:09'
@Desc : 'kaggle，根据twitter内容预测是否为真实的风险信息，预测'
"""

import numpy as np
import pandas as pd
import tensorflow.keras as tfk

if __name__ == '__main__':
    test_orig = pd.read_csv('./dataset/test.csv')
    test_set = np.load('./dataset/test_set.npy')

    test_orig_id = test_orig['id'].to_frame()

    model = tfk.models.load_model('./models/RealorNot_convandpool.h5')

    result_predict = model.predict(test_set)

    result_dataframe = pd.DataFrame(result_predict, columns=['target'])

    result_dataframe.loc[result_dataframe['target'] < 0.5, 'target'] = 0
    result_dataframe.loc[result_dataframe['target'] >= 0.5, 'target'] = 1
    result_dataframe['target'] = result_dataframe['target'].astype('int64')

    res_file = pd.concat([test_orig_id, result_dataframe], axis=1)
    res_file.to_csv('./models/test_predict_convandpool.csv', index=False)

    print('suc')
