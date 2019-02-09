# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     preprocession
   Description :
   date：          2018/12/23
-------------------------------------------------
   Change Activity:
                   2018/12/23:
-------------------------------------------------
"""

"""
文本预处理:
    - 读取语料，统一语料编码为utf-8
    - 断句，根据中文断句标识符实现分句，返回句子列表
    - 分词，去掉多余空格，去掉不需要的数字/符号/其他非汉字字符等，
      去掉停用词，是否依据词性去掉不必要的词依据情况而定
      返回二维列表，即句子列表中的元素是由每个句子的词组成的列表
"""

import jieba
import jieba.posseg

from initialVar import STOP_WORDS, SPEECH_CORPUS, SENTENCE_SEG


def load_stopwords(path=STOP_WORDS):

    with open(path, 'r') as f:
        lines = f.readlines()
    stop_words = [line.strip() for line in lines]

    return stop_words


def load_corpus(path=SPEECH_CORPUS):

    with open(path, 'r', encoding='utf-8') as f:
        txts = f.readlines()

    txt = " ".join([line.strip() for line in txts])

    return txt


def sentence_seg(text, seg_signal=SENTENCE_SEG):
    """
    :param text: a string
    :param seg_signal: a list
    :return: a list made up with sentences
    """
    text = [text, ]
    for seg in seg_signal:
        res, text = text, []
        for item in res:
            text += item.split(seg)

    result = [txt.strip() for txt in text if len(txt.strip()) > 0]

    return result


def word_seg(sentence, stop_words_path=STOP_WORDS):
    # 载入停用词
    stop_words = load_stopwords(path=stop_words_path)

    # 使用jieba分词
    jieba_result = jieba.cut(sentence)

    # # 如果基于词性筛选词的话，
    # jiaba_result = jieba.posseg.cut(sentence)
    #     # 返回的是个迭代器，其中的元素有两个属性(flag/word)
    # 如果要筛选数字或者字母的话，可以用string.isdigit()/string.isletters()
    # 如果仅匹配出中文字, 使用re.compile("^[\u4E00-\u9FFF]+$")即可

    # 这里不考虑词性筛选，仅过滤掉停用词
    word_list = [word.strip() for word in jieba_result
                 if word.strip() not in stop_words
                 and len(word.strip()) > 0]

    return word_list


def segment(text,
            sentence_seg_signal=SENTENCE_SEG,
            stop_words_path=STOP_WORDS):
    results = []
    sentences = sentence_seg(text, seg_signal=sentence_seg_signal)
    for sentence in sentences:
        word_list = word_seg(sentence, stop_words_path=stop_words_path)
        results.append(word_list)

    return results



if __name__ == '__main__':
    text = load_corpus()
    sentences = segment(text)
    print(sentences)





























