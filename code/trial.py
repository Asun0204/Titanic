#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/7/6
# Description:

import pandas as pd
submission_df = pd.read_csv('Data/gender_submission.csv', header=0)

value1 = submission_df.values
value2 = submission_df.as_matrix()
print value1.shape
print value2.shape
print submission_df.dtypes