#!/usr/bin/env python
from FlagEmbedding import FlagReranker
import utils


class Reranker:
    def __init__(self):
        self.reranker = FlagReranker(utils.get_reranker_path(), use_fp16=True, local_files_only=True)

    def __compute_reranking_score(self, query: str, passages: list[str]):
        return self.reranker.compute_score([(query, passage) for passage in passages], normalize=True)

    def do_rerank(self, query: str, passages: list[str], count=None) -> list[str]:
        scores = self.__compute_reranking_score(query, passages)
        result = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        # print(result)
        return [r[0] for r in result[:count]]
        # return result[:count]