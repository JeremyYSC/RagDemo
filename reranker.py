#!/usr/bin/env python
import pprint
from typing import Iterable
import time
# from FlagEmbedding import FlagReranker

from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
import utils


class Reranker:
    def __init__(self, model_path=None, top_n: int = None):
        if model_path is None:
            model_path = utils.get_reranker_path()
        # self.reranker = FlagReranker(model_path, use_fp16=True, local_files_only=True)
        self.reranker = OpenVINOReranker(model_name_or_path=model_path)
        self.top_n = top_n

    # def _compute_reranking_score(self, query: str, passages: list[str]):
    #     return self.reranker.compute_score([(query, passage) for passage in passages], normalize=True)

    def _compute_reranking_score(self, query: str, passages: list[str]):
        class Request(object):
            pass
        request = Request()
        request.query = query
        request.passages =  [
            {"id": i, "text": passage} for i, passage in enumerate(passages)
        ]
        return self.reranker.rerank(request)

    def do_rerank(self, question: str, passage_list: list, passage_getter=None) -> list:
        """
        :param question: received question
        :param passage_list: list of structure that contains passage, representing relevant chunks given by embedding
        :param passage_getter: way to get passage in the structured list, keep None while passing a list of pure passage
        :return: top_n most relevant results in passage_list
        """
        pprint.pprint('do_rerank')
        tmp = time.time()
        passages = passage_list if passage_getter is None else [passage_getter(element) for element in passage_list]
        scores = self._compute_reranking_score(question, passages)
        pprint.pprint(f"compute score: {time.time() - tmp:.3f}")
        tmp = time.time()
        result = sorted(zip(passage_list, scores), key=lambda x: x[1], reverse=True)
        pprint.pprint(f"sort: {time.time() - tmp:.3f}")
        return [r[0] for r in result]
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

    if __name__ == '__main__':
        from reranker import Reranker
        pprint.pprint("start")
        question = "What is panda?"
        passages = ["openvino tool kit panda panda", "my son is a dumb ass", "panda is a bear-liked animal",
                    "anda anda pp anda anda pp isis", "panda is using a cellphone", "pandas is a useful python toolkit",
                    "I want a panda"]
        reranker = Reranker(model_path=utils._get_model_path("bge-reranker-v2-m3-INT4"))
        pprint.pprint(reranker.do_rerank(question, passages))