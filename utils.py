# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   date：          2018/12/19
-------------------------------------------------
   Change Activity:
                   2018/12/19:
-------------------------------------------------
"""


import os
import math
import networkx as nx
import numpy as np
import sys


sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns',
                     'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

class AttriDict(dict):
    """Dict that can get attribute by dot"""
    def __init__(self, *args, **kwargs):
        super(AttriDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def as_text(text):
    """
    统一text的编码为utf-8
    :param text:
    :return:
    """
    return text


def debug(obj):
    pass


def get_similarity(word_list1, word_list2):
    """
    用于计算两个句子的相似度
    :param word_list1:
    :param word_list2:
    :return: similarity = (len(w for w in S1 and w in S2))
                        / (log(len(S1)) + log(len(S2)))
    """
    # Approach 1:
    words = list(set(word_list2 + word_list1))
    vec1 = [float(word_list1.count(word)) for word in words]
    vec2 = [float(word_list2.count(word)) for word in words]

    vec3 = [vec1[x] * vec2[x] for x in range(len(vec1))]
    vec4 = [1 for num in vec3 if num > 0.]
    co_occur_num = sum(vec4)

    # Approach 2:
    # S1 = len(word_list1)
    # S2 = len(word_list2)
    # if S1 <= S2:
    #     check_dict = dict.fromkeys(word_list1, 1)
    #     for word in word_list2:
    #         check_dict[word] = check_dict.get(word, -1)
    # else:
    #     check_dict = dict.fromkeys(word_list2, 1)
    #     for word in word_list1:
    #         check_dict[word] = check_dict.get(word, -1)
    # co_occur_num = len([word for word, freq in check_dict if freq > 0])

    if abs(co_occur_num) <= 1e-12:
        return 0

    denominator = math.log(float(len(word_list1))) + \
                math.log(float(len(word_list2)))

    if abs(denominator) < 1e-12:
        return 0

    return co_occur_num / denominator


def combine(word_list, window=2):
    """
    构造在window下的单词组合，用来构造单词之间的边
    :param word_list:
    :param window:
    :return:
    """
    window = max(2, window)
    for x in range(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r


def sort_words(vertext_source, edge_source,
               window=2, pagerank_config={'alpha': 0.85}):
    """
    将单词按关键程度从大到小排序
    :param vertext_source: 二维列表，用来构造pagerank中的节点
    :param edge_source: 二位列表，根据位置关系构造pagerank的边
    :param window: 一个句子中相邻的window个单词，两两之间认为有边
    :param pagerank_config: pangerank算法的超参数
    :return:
    """
    sorted_words = []
    word_index = {}
    index_word = {}
    _vertext_source = vertext_source
    _edge_source = edge_source
    words_number = 0
    for word_list in _vertext_source:
        for word in word_list:
            if word not in word_index:
                word_index[word] = words_number
                index_word[words_number] = word
                words_number += 1

    graph = np.zeros((words_number, words_number))

    for word_list in _edge_source:
        for w1, w2 in combine(word_list, window):
            if w1 in word_index and w2 in word_index:
                index1 = word_index[w1]
                index2 = word_index[w2]
                graph[index1][index2] = 1.0
                graph[index2][index1] = 1.0

    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)
        # This is a dict
    sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    for index, score in sorted_scores:
        item = AttriDict(word=index_word[index], weight=score)
        sorted_words.append(item)

    return sorted_words

def sort_sentences(sentences, words,
                   sim_func=get_similarity,
                   pagerank_config={'alpha':0.85}):
    """
    将句子按照关键程度从大到小排序
    :param sentences: 列表，元素是句子
    :param words: 二维列表，子列表和sentences中的句子对应，子列表由单词构成
    :param sim_func: 计算句子相似性，参数是两个由单词组成的列表
    :param pagerank_config: pagerank算法的超参数
    :return: 返回排好序的句子列表
    """
    sorted_sentences = []
    _source = words
    sentences_num = len(_source)
    graph = np.zeros((sentences_num, sentences_num))

    for x in range(sentences_num):
        for y in range(x, sentences_num):
            similarity = sim_func(_source[x], _source[y])
            graph[x, y] = similarity
            graph[y, x] = similarity

    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for index, score in sorted_scores:
        item = AttriDict(index=index,
                         sentence=sentences[index],
                         weight=score)
        sorted_sentences.append(item)

    return sorted_sentences























