#!/usr/bin/env python
from typing import Iterable

from FlagEmbedding import FlagReranker
import utils


class Reranker:
    def __init__(self, top_n: int = None):
        self.reranker = FlagReranker(utils.get_reranker_path(), use_fp16=True, local_files_only=True)
        self.top_n = top_n

    def _compute_reranking_score(self, query: str, passages: list[str]):
        return self.reranker.compute_score([(query, passage) for passage in passages], normalize=True)

    def do_rerank(self, question: str, passage_list: list, passage_getter=None) -> list:
        """
        :param question: received question
        :param passage_list: list of structure that contains passage, representing relevant chunks given by embedding
        :param passage_getter: way to get passage in the structured list, keep None while passing a list of pure passage
        :return: top_n most relevant results in passage_list
        """
        passages = passage_list if passage_getter is None else [passage_getter(element) for element in passage_list]
        scores = self._compute_reranking_score(question, passages)
        result = sorted(zip(passage_list, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in result[:self.top_n]]
        # return result[:self.top_n]

    def do_rerank_results(self, query: str, iterable_results: Iterable[tuple[str, dict, float]]) -> tuple[list[str], list[dict], list[float]]:
        results = list(iterable_results)
        passages = [item[0] for item in results]
        scores = self._compute_reranking_score(query, passages)
        # print("scores")
        # print(scores)

        sorted_results_with_scores = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        # print("sorted_results_with_scores")
        # print(sorted_results_with_scores)
        top_n_results_with_scores = sorted_results_with_scores[:self.top_n]
        top_n_results = [r[0] for r in top_n_results_with_scores]

        top_n_summaries, top_n_metadatas, top_n_distances = zip(*top_n_results)
        return list(top_n_summaries), list(top_n_metadatas), list(top_n_distances)