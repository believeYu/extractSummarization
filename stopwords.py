# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     stopwords
   Description :
   date：          2018/12/23
-------------------------------------------------
   Change Activity:
                   2018/12/23:
-------------------------------------------------
"""

"""
此文件不用
"""


import gzip
import json

file = 'data/stopwords.json.gz'
with gzip.open(file, 'rt', encoding='utf-8') as fp:
    _STOPWORDS = json.load(fp)

for i, j in _STOPWORDS.items():
    print(i, j)
    print()
























































