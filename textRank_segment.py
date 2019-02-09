# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     textRank_segment
   Description :
   date：          2018/12/20
-------------------------------------------------
   Change Activity:
                   2018/12/20:
-------------------------------------------------
"""

import jieba.posseg as pseg
import codecs
import utils
from utils import sort_words
from extract_wordfrequency import load_text

def get_default_stop_words_file():
    """

    :return: 返回停用词的文件路径
    """
    pass

class WordSegmentation:
    """分词"""

    def __init__(self,
                 stop_words_file=None,
                 allow_speech_tags=utils.allow_speech_tags):
        """
        :param stop_words_file: 停用词库
        :param allow_speech_tags: 词性列表，用于过滤
        """
        # 允许的词性标注符，解决编码问题
        # allow_speech_tags = [utils.as_text(item)
        #                      for item in allow_speech_tags]
        self.default_speetch_tag_filter = allow_speech_tags
        self.stop_words = set()
        # self.stop_words_file = get_default_stop_words_file() # 默认文件
        # if type(stop_words_file) is str: # 有提供则用提供的停用词库
        #     self.stop_words_file = stop_words_file
        # for word in codecs.open(self.stop_words_file, 'r', 'utf-8', 'ignore'):
        #     self.stop_words.add(word.strip())

    def segment(self,
                text,
                lower=True,
                use_stop_words=True,
                use_speech_tags_filter=False):
        """

        :param text: 被处理的文本对象
        :param lower: 是否小写
        :param use_stop_words: 若为True，则利用停用词来过滤
        :param use_speech_tags_filter: 是否基于词性进行过滤
        :return: 返回list类型的分词结果
        """
        text = utils.as_text(text)
        jieba_result = pseg.cut(text)
            # 返回一个迭代器，其中的元素有两个属性(flag/word)

        if use_speech_tags_filter == True:
            jieba_result = [w for w in jieba_result
                            if w.flag in self.default_speetch_tag_filter]
        else:
            jieba_result = [w for w in jieba_result]

        # 去掉特殊符号
        word_list = [w.word.strip() for w in jieba_result if w.flag != 'x']
        word_list = [word for word in word_list if len(word) > 0]

        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words:
            word_list = [word.strip() for word in word_list
                         if word.strip() not in self.stop_words]

        return word_list

    def segment_sentences(self, sentences,
                          lower=True,
                          use_stop_words=True,
                          use_speech_tags_filter=False):
        """
        将列表sentences中的每个元素/句子转换为由单词构成的列表
        :param sentences:
        :param lower:
        :param use_stop_words:
        :param use_speech_tags_filter:
        :return:
        """
        res = []
        for sentence in sentences:
            res.append(self.segment(
                text=sentence, lower=lower,
                use_stop_words=use_stop_words,
                use_speech_tags_filter=use_speech_tags_filter))
        return res


class SentenceSegmentation:
    """分句"""

    def __init__(self, delimiters=utils.sentence_delimiters):
        """

        :param delimiters: 用于拆分句子的标识符
        """
        self.delimiters = set([utils.as_text(item) for item in delimiters])
            # 统一标识符的编码

    def segment(self, text):
        res = [utils.as_text(text)]
            # 统一文本编码
        utils.debug(res)
        utils.debug(self.delimiters)

        for sep in self.delimiters:
            # 这段分句代码，技巧很高，但(理论上)时间复杂度是指数级别的
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
            # 去掉空格符
        return res


class Segmentation:

    def __init__(self, stop_words_file=None,
                 allow_speech_tags=utils.allow_speech_tags,
                 delimites=utils.sentence_delimiters):
        """

        :param stop_words_file: 停用词
        :param allow_speech_tags: 词性筛选
        :param delimites: 句子分隔符
        """
        self.ws = WordSegmentation(
            stop_words_file=stop_words_file,
            allow_speech_tags=allow_speech_tags)
        self.ss = SentenceSegmentation(delimiters=delimites)

    def segment(self, text, lower=False):
        text = utils.as_text(text)

        sentences = self.ss.segment(text)

        words_no_filter = self.ws.segment_sentences(
            sentences=sentences, lower=lower,
            use_stop_words=False,
            use_speech_tags_filter=False)

        words_no_stop_words = self.ws.segment_sentences(
            sentences=sentences, lower=lower,
            use_stop_words=True,
            use_speech_tags_filter=False)

        words_all_filters = self.ws.segment_sentences(
            sentences=sentences, lower=lower,
            use_stop_words=True,
            use_speech_tags_filter=True)

        return utils.AttriDict(
            sentences = sentences,
            words_no_filter = words_no_filter,
            words_no_stop_words = words_no_stop_words,
            words_all_filters = words_all_filters)


if __name__ == '__main__':
    text = load_text()
    segmen = Segmentation()
    result = segmen.segment(text).words_no_filter
    # print(result)
    sort_words(result)





















































