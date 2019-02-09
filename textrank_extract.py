# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     textrank_extract
   Description :
   date：          2018/12/24
-------------------------------------------------
   Change Activity:
                   2018/12/24:
-------------------------------------------------
"""

"""
说明:
    - textRank算法是基于图结构的算法，类似PageRank算法计算网页的重要性，
      它以同样的思路计算句子的重要性.
    - 其设计思路是:
        -1. 构建图，图节点是句子，边代表的是句子的相似度;
            相似度计算公式: similarity = (len(w for w in S1 and w in S2))
                        / (log(len(S1)) + log(len(S2)))
            即，看两个句子的共有词来判断句子相似程度;
            
        -2. 有了图和边，便有了转移矩阵M，依据以下公式计算每个句子的PR值:
            pr_vector = (1-d) + d * M * pr_vector_last (d=0.85)
            
        -3. 最后依据PR值选出最高得分的句子，按照行文顺序输出
    - 两点补充:
        -1. lexrank算法与之类似，同样是基于图结构的算法;
            不同在于，句子相似关系计算方法不同.
            
            句子间的相似关系由余弦相似度计算，句子表征向量为tfidf值;
            
            关键语句的选择取决于评分标准(即类似pagerank的PR值)，
            要充分考虑每个句子对应节点的连线数量和粗细，
            最终选出得分高的节点作为关键语句.
            
        -2. 为了避免排名靠前的几句话表达相似重复的意思，
            可以将句子相似度做惩罚项加入评分中.
            a * score(i) + (1-a) * similarity(i, i-1), i = 2, 3, ..., N
            序号i表示排序后的顺序，从第二句开始，后面的句子必须和前一句进行相似度惩罚。
            这就是MMR(Maximum Margin Relevance)算法。
    - 这里仅提供textrank算法的代码        
"""


import math
import networkx as nx
import numpy as np

from preprocession import segment, sentence_seg, load_corpus





class TextRank:

    def __init__(self, text):
        # 将文本处理为一维的句子列表
        self.sentences = sentence_seg(text)

        # 二维的句子列表，元素是词
        self.words = segment(text)


    def get_similarity(self, word_list1, word_list2):
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


    def sort_sentences(self,
                       pagerank_config={'alpha':0.85}):
        """
        将句子按照关键程度从大到小排序
        :param pagerank_config: pagerank算法的超参数
        :return: 返回排好序的句子列表
        """
        # 用于保存最终结果
        sorted_sentences = []

        _source = self.words
        sentences_num = len(_source)

        # 构建词的两两对应的矩阵，即图的边，边代表句子间的相似度
        graph = np.zeros((sentences_num, sentences_num))
        for x in range(sentences_num):
            for y in range(x, sentences_num):
                similarity = self.get_similarity(_source[x], _source[y])
                graph[x, y] = similarity
                graph[y, x] = similarity

        # networkx库专用于构建图结构，能调用API直接计算PR值
        nx_graph = nx.from_numpy_matrix(graph)
        scores = nx.pagerank(nx_graph, **pagerank_config)

        # 依据计算结果对词排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for index, score in sorted_scores:
            item = dict(index=index,
                        sentence=self.sentences[index],
                        weight=score)
            sorted_sentences.append(item)

        return sorted_sentences


    def get_key_sentences(self, num=6, sentence_min_len=6):
        """
        获取最重要的num个长度大于sentence_min_len的句子用来生成摘要
        :param num:
        :param sentence_min_len:
        :return:
        """
        key_sentences = self.sort_sentences()
        result = []
        count = 0
        for item in key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                result.append(item)
                count += 1
        return result


    def summarization(self):

        result = self.get_key_sentences()
        final_res = [(item['sentence'], item['index']) for item in result]
        finale_res = list(sorted(final_res, key=lambda x: x[1], reverse=False))
        return [item[0] for item in finale_res]


if __name__ == '__main__':
    text = load_corpus()
    textrank = TextRank(text)
    print(textrank.summarization())


































