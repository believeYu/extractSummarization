# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     textRank_kword
   Description :
   date：          2018/12/20
-------------------------------------------------
   Change Activity:
                   2018/12/20:
-------------------------------------------------
"""

import networkx as nx
import numpy as np

from . import utils
from .textRank_segment import Segmentation



class TextRankKeyword:

    def __init__(self, stop_words_file=None,
               allow_speech_tags=utils.allow_speech_tags,
               delimiters=utils.sentence_delimiters):
        self.text = ""
        self.keywords = None

        self.seg = Segmentation(
            stop_words_file=stop_words_file,
            allow_speech_tags=allow_speech_tags,
            delimites=delimiters)

        self.sentences = None
        self.words_no_filter = None
        self.words_all_filters = None

    def analyze(self, text,
                window=2,
                lower=False,
                vertex_source='all_filters',
                edge_source='no_stope_words',
                pagerank_config={'alpha': 0.85}):
        """分析文本
        Keyword arguments:
        text       --  文本内容，字符串。
        window     --  窗口大小，int，用来构造单词之间的边。默认值为2。
        lower      --  是否将文本转换为小写。默认为False。
        vertex_source   --  选择使用words_no_filter, words_no_stop_words,
            words_all_filters中的哪一个来构造pagerank对应的图中的节点。
            默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words',
            'all_filters'`。关键词也来自`vertex_source`。
        edge_source     --  选择使用words_no_filter, words_no_stop_words,
            words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边。
            默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words',
            'all_filters'`。边的构造要结合`window`参数。
        """
        # self.text = util.as_text(text)
        self.text = text
        self.word_index = {}
        self.index_word = {}
        self.keywords = []
        self.graph = None

        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_all_filters = result.words_all_filters

        options = ['no_filter', 'no_stop_words', 'all_filters']

        if vertex_source in options:
            _vertex_source = result['words_' + vertex_source]
        else:
            _vertex_source = result['words_all_filters']

        if edge_source in options:
            _edge_source = result['words_' + edge_source]
        else:
            _edge_source = result['words_no_stop_words']

        self.keywords = utils.sort_words(
            _vertex_source, _edge_source,
            window=window, pagerank_config=pagerank_config)

    def get_keywords(self, num=6, word_min_len=1):
        """
        获取最重要的num个长度不低于word_min_len的关键词
        :param num:
        :param word_min_len:
        :return: 返回关键词列表
        """
        result = []
        count = 0
        for item in self.keywords:
            if count >= num:
                break
            if len(item.word) >= word_min_len:
                result.append(item)
                count += 1
        return result

    def get_keyphrases(self, keywords_num=12, min_occur_num=2):
        """
        获取keywords_num个关键短语，要求，这个短语出现次数不低于min_occur_num
        :param keywords_num:
        :param min_occur_num:
        :return:
        """
        keywords_set = set([item.word for item in self.get_keywords(
            num=keywords_num, word_min_len=1)])
        keyphrases = set()
        for sentence in self.words_no_filter:
            one = []
            for word in sentence:
                if word in keywords_set:
                    one.append(word)
                else:
                    if len(one) > 1:
                        keyphrases.add("".join(one))
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            if len(one) > 1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases
                if self.text.count(phrase) >= min_occur_num]




































