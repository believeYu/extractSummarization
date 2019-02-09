# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     textRank_sentence
   Description :
   date：          2018/12/21
-------------------------------------------------
   Change Activity:
                   2018/12/21:
-------------------------------------------------
"""

import networkx as nx
import numpy as np
import utils
from textRank_segment import Segmentation
from extract_wordfrequency import load_text


class TextRank4Sentence(object):

    def __init__(self, stop_words_file=None,
                 allow_speech_tags=utils.allow_speech_tags,
                 delimiters=utils.sentence_delimiters):
        """
        文本过滤参数
        :param stop_words_file:
        :param allow_speech_tags:
        :param delimiters:
        """
        self.seg = Segmentation(
            stop_words_file=stop_words_file,
            allow_speech_tags=allow_speech_tags,
            delimites=delimiters)
        self.sentences = None
        self.words_no_filter = None
        self.key_sentences = None

    def analyze(self, text,
                lower=False,
                source='no_filter',
                sim_func=utils.get_similarity,
                pagerank_config={'alpha': 0.85}):

        self.key_sentences = []
        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        _source = result['words_' + source]

        self.key_sentences = utils.sort_sentences(
            sentences=self.sentences, words=_source)

    def get_key_sentences(self, num=6, sentence_min_len=6):
        """
        获取最重要的num个长度大于sentence_min_len的句子用来生成摘要
        :param num:
        :param sentence_min_len:
        :return:
        """
        result = []
        count = 0
        for item in self.key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                result.append(item)
                count += 1
        return result


if __name__ == '__main__':
    text = load_text()
    print(text)
    print()
    abstract = TextRank4Sentence()
    abstract.analyze(text)
    result = abstract.get_key_sentences()
    final_res = [(item['sentence'], item['index']) for item in result]
    finale_res = list(sorted(final_res, key=lambda x:x[1], reverse=False))
    print([item[0] for item in finale_res])






























































