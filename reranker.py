#!/usr/bin/env python
import os
from FlagEmbedding import FlagReranker


class Reranker:
    MODEL_PATH = "model"
    RERANKER_NAME = "bge-reranker-v2-m3"
    def __init__(self):
        path = os.path.join(self.MODEL_PATH, self.RERANKER_NAME)
        self.reranker = FlagReranker(path, use_fp16=True, local_files_only=True)

    def __compute_reranking_score(self, query: str, passages: list[str]):
        return self.reranker.compute_score([(query, passage) for passage in passages], normalize=True)

    def do_rerank(self, query: str, passages: list[str], count=None) -> list[str]:
        scores = self.__compute_reranking_score(query, passages)
        result = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        # print(result)
        return [r[0] for r in result[:count]]
        # return result[:count]