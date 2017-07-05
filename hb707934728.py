#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/6/27
# Description: 利用随机森林,xgboost,logistic回归,预测泰坦尼克号上面的乘客的获救概率
# Site: http://blog.csdn.net/hb707934728/article/details/70738778

import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv


def show_accuracy(a, b, tip):
    """计算预测准确率

    :param a: 预测值
    :param b: 实际值
    :param tip: not used
    :return: 准确率*100
    """
    # 展平成一维数组，和flatten类似
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    return acc_rate


def load_data(file_name, is_train):
    """加载数据和数据预处理

    :param file_name: 数据文件的位置
    :param is_train: 标记是否为训练集
    :return: 训练集返回x和标签y，测试集返回x和乘客ID
    """
    data = pd.read_csv(file_name)  # 数据文件路径
    # print 'data.describe() = \n', data.describe()

    # 性别 将性别字段Sex中的值 female用0，male用1代替,类型 int
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # 补齐船票价格缺失值
    if len(data.Fare[data.Fare.isnull()]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median()
        for f in range(0, 3):  # loop 0 to 2
            data.loc[(data.Fare.isnull()) & (data.Pclass == f + 1), 'Fare'] = fare[f]

    # 年龄：使用均值代替缺失值
    # mean_age = data['Age'].dropna().mean()
    # data.loc[(data.Age.isnull()), 'Age'] = mean_age
    if is_train:
        # 年龄：使用随机森林预测年龄缺失值
        print '随机森林预测缺失年龄：--start--'
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        # print 'data_for_age=\n', data_for_age
        # print 'age_exis=\n', age_exist
        # print 'age_null=\n',age_null
        print age_exist
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        # print 'x = age_exist.values[:, 1:] 中 x=',x
        # print 'y = age_exist.values[:, 0] 中 y=',y
        # n_estimators 决策树的个数，越多越好,值越大，性能就会越差,但至少100
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        # print 'age_hat',age_hat
        # 填充年龄字段中值为空的
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print '随机森林预测缺失年龄：--over--'
    else:
        print '随机森林预测缺失年龄2：--start--'
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        # print age_exist
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print '随机森林预测缺失年龄2：--over--'

        # 起始城市
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'  # 保留缺失出发城市
    # print data['Embarked']
    embarked_data = pd.get_dummies(data.Embarked)
    # print embarked_data
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)
    # print data.describe()
    data.to_csv('Data/New_Data.csv')

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    # x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = None
    if 'Survived' in data:
        y = data['Survived']

    x = np.array(x)
    y = np.array(y)

    x = np.tile(x, (5, 1))
    y = np.tile(y, (5,))
    if is_train:
        return x, y
    return x, data['PassengerId']


def write_result(c, c_type):
    file_name = 'Data/test.csv'
    x, passenger_id = load_data(file_name, False)

    if type == 3:
        x = xgb.DMatrix(x)
    y = c.predict(x)
    y[y > 0.5] = 1
    y[~(y > 0.5)] = 0

    predictions_file = open("Prediction_%d.csv" % c_type, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", "Survived"])
    open_file_object.writerows(zip(passenger_id, y))
    predictions_file.close()


def totalSurvival(y_hat, tip):
    total = 0
    for index, value in enumerate(y_hat):
        if value == 1:
            total = total + 1
    print tip, '存活：', total
    print '人'


if __name__ == "__main__":
    # 加载并完善特征数据
    x, y = load_data('Data/train.csv', True)
    # 划分训练集和测试集x表示样本特征集，y表示样本结果  test_size 样本占比,random_state 随机数的种子
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

    # print 'x_train=',x_train,'y_train=',y_train

    # logistic回归
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    y_hat = lr.predict(x_test)
    lr_rate = show_accuracy(y_hat, y_test, 'Logistic回归 ')
    totalSurvival(y_hat, 'Logistic回归')
    # 随机森林 n_estimators：决策树的个数,越多越好,不过值越大，性能就会越差,至少100
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_hat = rfc.predict(x_test)
    rfc_rate = show_accuracy(y_hat, y_test, '随机森林 ')
    totalSurvival(y_hat, '随机森林')
    # write_result(rfc, 2)

    # XGBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
    y_hat = bst.predict(data_test)
    y_hat[y_hat > 0.5] = 1
    y_hat[~(y_hat > 0.5)] = 0
    xgb_rate = show_accuracy(y_hat, y_test, 'XGBoost ')
    totalSurvival(y_hat, 'xgboost')

    print 'Logistic回归：%.3f%%' % lr_rate
    print '随机森林: %.3f%%' % rfc_rate
    print 'XGBoost: %.3f%%' % xgb_rate