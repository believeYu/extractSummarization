# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     bm25
   Description :
   date：          2018/12/16
-------------------------------------------------
   Change Activity:
                   2018/12/16:
-------------------------------------------------
"""

import math


class BM25():
    """计算文本相似度，较tf-idf有所改进"""
    def __init__(self, docs):
        self.D = len(docs)
        self.avgd1 = sum([len(doc) + 0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []
        self.df = {}
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs: # doc是个句子
            # Count the word frequency
            temp = {}
            for word in doc:
                # if word not in temp:
                #     temp[word] = 0
                # temp[word] += 1
                temp[word] = temp.get(word, 0) + 1
                self.df[word] = self.df.get(word, 0) + 1
            self.f.append(temp)

        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] +
                         self.k1 * (1 - self.b + self.b * d / self.avgd1)))

    def simall(self, doc):
        # scores = []
        # for index in range(self.D):
        #     score = self.sim(doc, index)
        #     scores.append(score)
        scores = [self.sim(doc, index) for index in range(self.D)]
        return scores










































