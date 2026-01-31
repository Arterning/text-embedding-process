#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于BM25+TextRank的中文文章自动摘要提取
"""

import re
import jieba
import jieba.posseg as pseg
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi
import networkx as nx


class ChineseTextSummarizer:
    """中文文本摘要提取器"""

    def __init__(self, stop_words=None):
        """
        初始化摘要提取器

        Args:
            stop_words: 停用词列表，如果为None则使用默认停用词
        """
        self.stop_words = stop_words or self._get_default_stopwords()

    def _get_default_stopwords(self):
        """获取默认停用词"""
        return set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '那', '为', '以', '与', '之', '于', '对', '把',
            '从', '被', '跟', '给', '向', '往', '得', '地', '他', '她', '它', '们'
        ])

    def _sentence_split(self, text):
        """
        分句

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 使用标点符号分句
        text = re.sub(r'\n+', '。', text)
        sentences = re.split(r'[。！？；\n]', text)
        # 过滤掉空句子和过短的句子
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        return sentences

    def _tokenize(self, sentence):
        """
        中文分词并去除停用词

        Args:
            sentence: 输入句子

        Returns:
            分词后的词列表
        """
        words = jieba.lcut(sentence)
        # 去除停用词和单字符词
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        return words

    def _calculate_bm25_scores(self, sentences, sentences_words):
        """
        使用BM25计算句子与文档的相关性得分

        Args:
            sentences: 原始句子列表
            sentences_words: 分词后的句子列表

        Returns:
            BM25得分列表
        """
        # 初始化BM25
        bm25 = BM25Okapi(sentences_words)

        # 将所有句子的词合并作为查询
        all_words = []
        for words in sentences_words:
            all_words.extend(words)

        # 计算每个句子的BM25得分
        bm25_scores = bm25.get_scores(all_words)

        return np.array(bm25_scores)

    def _calculate_sentence_similarity(self, sent1_words, sent2_words):
        """
        计算两个句子的相似度

        Args:
            sent1_words: 句子1的分词列表
            sent2_words: 句子2的分词列表

        Returns:
            相似度得分
        """
        if not sent1_words or not sent2_words:
            return 0.0

        # 使用Jaccard相似度
        set1 = set(sent1_words)
        set2 = set(sent2_words)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_textrank_scores(self, sentences_words):
        """
        使用TextRank算法计算句子重要性得分

        Args:
            sentences_words: 分词后的句子列表

        Returns:
            TextRank得分列表
        """
        # 构建句子相似度图
        n = len(sentences_words)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._calculate_sentence_similarity(
                    sentences_words[i],
                    sentences_words[j]
                )
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim

        # 使用NetworkX计算PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, max_iter=100)

        # 转换为列表
        textrank_scores = [scores[i] for i in range(n)]

        return np.array(textrank_scores)

    def summarize(self, text, ratio=0.3, top_n=None, bm25_weight=0.5):
        """
        生成文本摘要

        Args:
            text: 输入文本字符串
            ratio: 摘要比例（0-1之间），默认0.3表示提取30%的句子
            top_n: 提取的句子数量，如果指定则忽略ratio
            bm25_weight: BM25得分的权重（0-1之间），TextRank权重为1-bm25_weight

        Returns:
            摘要文本字符串
        """
        # 分句
        sentences = self._sentence_split(text)

        if len(sentences) == 0:
            return ""

        if len(sentences) == 1:
            return sentences[0]

        # 分词
        sentences_words = [self._tokenize(sent) for sent in sentences]

        # 过滤掉没有有效词的句子
        valid_indices = [i for i, words in enumerate(sentences_words) if len(words) > 0]
        if not valid_indices:
            return sentences[0]

        sentences = [sentences[i] for i in valid_indices]
        sentences_words = [sentences_words[i] for i in valid_indices]

        # 计算BM25得分
        bm25_scores = self._calculate_bm25_scores(sentences, sentences_words)

        # 计算TextRank得分
        textrank_scores = self._calculate_textrank_scores(sentences_words)

        # 归一化得分
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        if textrank_scores.max() > 0:
            textrank_scores = textrank_scores / textrank_scores.max()

        # 组合得分
        combined_scores = bm25_weight * bm25_scores + (1 - bm25_weight) * textrank_scores

        # 确定提取的句子数量
        if top_n is None:
            top_n = max(1, int(len(sentences) * ratio))
        else:
            top_n = min(top_n, len(sentences))

        # 选择得分最高的句子
        top_indices = np.argsort(combined_scores)[-top_n:]

        # 按原文顺序排序
        top_indices = sorted(top_indices)

        # 生成摘要
        summary_sentences = [sentences[i] for i in top_indices]
        summary = '。'.join(summary_sentences) + '。'

        return summary

    def get_sentence_scores(self, text, bm25_weight=0.5):
        """
        获取每个句子的得分详情

        Args:
            text: 输入文本字符串
            bm25_weight: BM25得分的权重

        Returns:
            包含句子及其得分的列表
        """
        sentences = self._sentence_split(text)

        if len(sentences) == 0:
            return []

        sentences_words = [self._tokenize(sent) for sent in sentences]

        valid_indices = [i for i, words in enumerate(sentences_words) if len(words) > 0]
        if not valid_indices:
            return [(sentences[0], 1.0, 0.0, 0.0, 1.0)]

        sentences = [sentences[i] for i in valid_indices]
        sentences_words = [sentences_words[i] for i in valid_indices]

        bm25_scores = self._calculate_bm25_scores(sentences, sentences_words)
        textrank_scores = self._calculate_textrank_scores(sentences_words)

        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        if textrank_scores.max() > 0:
            textrank_scores = textrank_scores / textrank_scores.max()

        combined_scores = bm25_weight * bm25_scores + (1 - bm25_weight) * textrank_scores

        results = []
        for i, sent in enumerate(sentences):
            results.append({
                'sentence': sent,
                'bm25_score': float(bm25_scores[i]),
                'textrank_score': float(textrank_scores[i]),
                'combined_score': float(combined_scores[i])
            })

        return results


def main():
    """示例用法"""

    # 示例文本
    sample_text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。
    人工智能可以对人的意识、思维的信息过程进行模拟。
    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    人工智能是一门极富挑战性的科学，从事这项工作的人必须懂得计算机知识、心理学和哲学。
    人工智能是包括十分广泛的科学，它由不同的领域组成，如机器学习、计算机视觉等等。
    总的说来，人工智能研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。
    """

    # 创建摘要器
    summarizer = ChineseTextSummarizer()

    # 生成摘要
    print("=" * 50)
    print("原文：")
    print(sample_text.strip())
    print("\n" + "=" * 50)

    # 提取30%的句子作为摘要
    summary = summarizer.summarize(sample_text, ratio=0.3)
    print("摘要（ratio=0.3）：")
    print(summary)
    print("\n" + "=" * 50)

    # 提取固定数量的句子
    summary = summarizer.summarize(sample_text, top_n=3)
    print("摘要（top_n=3）：")
    print(summary)
    print("\n" + "=" * 50)

    # 查看每个句子的得分
    print("句子得分详情：")
    scores = summarizer.get_sentence_scores(sample_text)
    for i, score_info in enumerate(scores, 1):
        print(f"\n句子{i}: {score_info['sentence']}")
        print(f"  BM25得分: {score_info['bm25_score']:.4f}")
        print(f"  TextRank得分: {score_info['textrank_score']:.4f}")
        print(f"  综合得分: {score_info['combined_score']:.4f}")


if __name__ == "__main__":
    main()
