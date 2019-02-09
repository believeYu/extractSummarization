# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     wordFreq_extract
   Description :
   date：          2018/12/23
-------------------------------------------------
   Change Activity:
                   2018/12/23:
-------------------------------------------------
"""

"""
说明:
    - 抽取式生成文本摘要的系列算法最重要的假设是，
      认为文章有些核心语句可以涵盖全篇的重要信息;
    - 算法的任务就是找出那些涵盖重要信息的语句。
    - 当把文章切分为句子列表后，整个任务其实是个排序问题，排序靠前的句子，
      自然就是涵盖重要信息的句子。
    - 区别在于如何对句子排序
    - 这里的排序算法设计如下: 
        1. 简单统计文档内部每个词的tfidf值，选出前50个tfidf值高的词
        2. 一个句子的词列表中，以3到5个词为一簇(即包含关键词的句子片段);
           给每个簇打分，公式是 score = (num of keywords) ** 2 / length of cluster
        3. 找出含有最高分簇的句子，比如5句或者更多(少)
        4. 按照文章的行文顺序，对句子进行排列输出，即生成文本摘要       
"""



import math
import nltk
from preprocession import segment, load_corpus, sentence_seg


# 设置三个超参数
# 前topn个高频词
TOP_WORD = 60

# 簇的长度
CLUSTER_THRESHOLD = 5

# 含有最高分簇的句子中, 返回前n个句子
TOP_SENTENCE = 5


class TfIdf:

    def __init__(self, sentences):
        """

        :param sentences: 二维列表，句子列表，每个句子由词列表组成
        """
        self.sentences = sentences
        self.words = []
        for sentence in self.sentences:
            self.words += sentence

    def tf_value(self):
        word_tf = nltk.FreqDist(self.words)
        return word_tf

    def idf_value(self):
        word_idf = {}

        # 总句子数
        tt_count = len(self.sentences)

        # 每个词出现的句子数
        for sentence in sentences:
            for word in set(sentence):
                word_idf[word] = word_idf.get(word, 0.0) + 1.0

        # 按公式转换为idf值
        for w, v in word_idf.items():
            word_idf[w] = math.log(tt_count / (1.0 + v))

        return word_idf

    def get_tfidf(self):
        word_tfidf = {}

        word_tf = self.tf_value()
        word_idf = self.idf_value()

        for word, idf in word_idf.items():
            tf = word_tf.get(word, 1.0)
            tf_idf = tf * idf
            word_tfidf[word] = tf_idf

        return word_tfidf

    def get_topn(self, n=TOP_WORD):
        word_tfidf = self.get_tfidf()
        word_sort = sorted(
            word_tfidf.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in word_sort][:n]


def score_sentences(sentences, topn_words,
                    threshold=CLUSTER_THRESHOLD):
    """
    :param sentences: 二维列表，句子列表，每个句子由词列表组成
    :param topn_words:
    :return:
    """
    scores = []
    sentence_idx = -1
    for s in sentences:
        # 记录句子的行文顺序
        sentence_idx += 1

        # 记录每个高频词在当前句子s中的索引位置
        word_idx = []
        for w in topn_words:
            try:
                word_idx.append(s.index(w))
            except ValueError:  # w不在句子中
                pass
        word_idx.sort()

        if len(word_idx) == 0:  # 如果当前句子并不包含高频词，跳过
            continue

        # 找到当前句子的簇
        clusters = []
        cluster = [word_idx[0]]
        i = 1
        while i < len(word_idx):
            if word_idx[i] - word_idx[i - 1] < threshold:
                cluster.append(word_idx[i])
            else:
                clusters.append(cluster[:])
                cluster = [word_idx[i]]
            i += 1
        clusters.append(cluster)

        # 计算句子的簇得分
        max_cluster_score = 0
        for c in clusters:
            significant_words_in_cluster = len(c)
            total_words_in_cluster = c[-1] - c[0] + 1
            score = 1.0 * significant_words_in_cluster ** 2 \
                    / total_words_in_cluster
            if score > max_cluster_score:
                max_cluster_score = score
        scores.append((sentence_idx, max_cluster_score))

    return scores


def summarization(text,
                  top_word=TOP_WORD,
                  cluster_threshold=CLUSTER_THRESHOLD,
                  top_sentence=TOP_SENTENCE):
    # 生成一维句子列表，以便还原摘要
    sentence_list = sentence_seg(text)
    # 生成二维句子列表，方便后续处理
    sentences = segment(text)

    # 得到每个词的tfidf值，从而得到关键词
    tfidf = TfIdf(sentences)
    topn_words = tfidf.get_topn(n=top_word)

    # 依据关键词计算每个句子的得分
    scored_sentence = score_sentences(
        sentences, topn_words=topn_words, threshold=cluster_threshold)

    top_n_scored = sorted(
        scored_sentence, key=lambda s: s[1])[-top_sentence:]

    return [sentence_list[idx] for idx, score in top_n_scored]



if __name__ == '__main__':
    text = load_corpus()
    sentences = segment(text)
    tfidf = TfIdf(sentences)
    topn_words = tfidf.get_topn()
    print(summarization(text))

















































