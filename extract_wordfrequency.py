# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     extract_wordfrequency
   Description :
   date：          2018/12/16
-------------------------------------------------
   Change Activity:
                   2018/12/16:
-------------------------------------------------
"""

import re
import nltk
import numpy as np
import jieba
import codecs
import zhon.hanzi as hanzi



N = 50 # 单词数量
CLUSTER_THRESHOLD = 15 # 单词间的距离
TOP_SENTENCES = 5 # 返回的 top n句子

# 分句
def sent_tokensizer(txts):
    pattern = ["。", "！", "？", " ", ".", "!", "?"]
    sentences = []
    sentence = ""
    for s in txts:
        sentence += s
        if s in pattern:
            # print("\n", sentence)
            sentences.append(sentence)
            sentence = ""
    return sentences


# 停用词
def load_stopwords(path='data/chinese_stopwords.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
    stop_words = [line.strip() for line in lines]

    return stop_words


# 摘要
def summarize(text):
    # stopwords = load_stopwords('')
    # sentences = sent_tokensizer(text) # 分句子
    # words = [w for sentence in sentences for w in jieba.cut(sentence)
    #          if w not in stopwords if len(w) > 1 and w != '\t'] # 分词
    # wordfre = nltk.FreqDist(words) # 统计词频
    # topn_words = [w[0] for w in
    #               sorted(wordfre.items(), key=lambda d:d[1], reverse=True)][:N]
    #             # 取出前100个高频词
    sentences = sent_tokensizer(text)
    words = [w for sentence in sentences for w in jieba.cut(sentence)
             if len(w) > 4 and w != '\t']
    wordFreq = nltk.FreqDist(words)
    # 提取前100个高频词
    topn_words = [w[0] for w in
                 sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)][:N]

    # 依据高频词计算句子得分
    scored_sentences = score_sentences(sentences, topn_words)

    #1.利用均值和标准差过滤非重要句子
    avg = np.mean([s[1] for s in scored_sentences])
    std = np.std([s[1] for s in scored_sentences])
    mean_scored = [(sent_idx, score) for (sent_idx, score)
                   in scored_sentences if score > (avg + 0.6 * std)]

    #2. 返回top n句子
    top_n_scored = sorted(
        scored_sentences, key=lambda s: s[1])[-TOP_SENTENCES:]
    top_n_scored = sorted(top_n_scored, key=lambda s: s[0])

    return dict(
        topn_summary=[sentences[idx] for (idx, score) in top_n_scored],
        mean_summary=[sentences[idx] for (idx, score) in mean_scored])

def score_sentences(sentences, topn_words):
    scores = []
    sentence_idx = -1
    for s in [list(jieba.cut(s)) for s in sentences]:
        sentence_idx += 1
        word_idx = []
        for w in topn_words:
            try:
                word_idx.append(s.index(w))
            except ValueError: # w不在句子中
                pass
        word_idx.sort()
        if len(word_idx) == 0:
            continue

        clusters = []
        cluster = [word_idx[0]]
        i = 1
        while i < len(word_idx):
            if word_idx[i] - word_idx[i - 1] < CLUSTER_THRESHOLD:
                cluster.append(word_idx[i])
            else:
                clusters.append(cluster[:])
                cluster = [word_idx[i]]
            i += 1

        clusters.append(cluster)
        max_cluster_score = 0
        for c in clusters:
            significant_words_in_cluster = len(c)
            total_words_in_cluster = c[-1] - c[0] + 1
            score = 1.0 * significant_words_in_cluster**2 \
                    / total_words_in_cluster
            if score > max_cluster_score:
                max_cluster_score = score
        scores.append((sentence_idx, max_cluster_score))

    return scores

def load_text(path="news_corpus.txt"):
    with open(path, 'r', encoding='utf-8') as f:
        txts = f.readlines()

    txt = ""
    for line in txts:
        if line != "\n":
            txt += line.strip()

    return txt


if __name__ == '__main__':
    # txts = load_text()
    # result = summarize(txts)
    # print('approch1: ', result['topn_summary'])
    # print()
    # print('approch2: ', result['mean_summary'])

    txts2s = load_text(path='speech_corpus.txt')
    # res = summarize(txts2s)
    # print('approch1: ', res['topn_summary'])
    # print()
    # print('approch2: ', res['mean_summary'])

    sentences = sent_tokensizer(txts2s)
    words = [w for sentence in sentences for w in jieba.cut(sentence)
             if len(w) > 1 and w != '\t']
    wordsFreq = nltk.FreqDist(words)
    topn_word = sorted(wordsFreq.items(), key=lambda x: x[1], reverse=True)[:100]
    print(topn_word)



































































