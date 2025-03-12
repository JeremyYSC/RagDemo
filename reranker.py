#!/usr/bin/env python
from FlagEmbedding import FlagReranker
import utils


class Reranker:
    def __init__(self, top_n: int = None):
        self.reranker = FlagReranker(utils.get_reranker_path(), use_fp16=True, local_files_only=True)
        self.top_n = top_n

    def __compute_reranking_score(self, query: str, passages: list[str]):
        return self.reranker.compute_score([(query, passage) for passage in passages], normalize=True)

    def do_rerank(self, query: str, passages: list[str]) -> list[str]:
        """
        :param query: query given by user
        :param passages: relevant chunks given by embedding
        :return: top_n most relevant result
        """
        scores = self.__compute_reranking_score(query, passages)
        result = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        # print(result)
        return [r[0] for r in result[:self.top_n]]
        # return result[:self.top_n]