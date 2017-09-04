#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/7/5
# Description: 使用分类决策树来进行分类

import pandas as pd
# import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

train_df = pd.read_csv('Data/train.csv', header=0)
test_df = pd.read_csv('Data/test.csv', header=0)
submission_df = pd.read_csv('Data/gender_submission.csv', header=0)

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        # 如果数据类型是object，就选出频率最大的作为缺失值，否则就选中位数
        # fill保存各个特征填充的缺失值，返回DataFrame类型
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].median() for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns = ['Sex']

# 把训练集和测试集的数据合在一起然后去处理缺失值，避免分别处理缺失值导致训练集和测试集的分布不同
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# 量化非数值型的数据，比如说male和female转化为1和0
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# 把处理好的数据集分成训练集和测试集
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

# 模型训练和预测
# gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
# predictions = gbm.predict(test_X)
# 决策树分类
dtc = DecisionTreeClassifier(random_state=0, criterion='gini', max_leaf_nodes=10)
dtc.fit(train_X, train_y)
predictions = dtc.predict(test_X)

# 把预测结果输出到当前目录的submission.csv文件中
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)

print 'Score:%.2f' % dtc.score(test_X, submission_df.as_matrix()[:, 1])