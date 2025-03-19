#!/usr/bin/env python
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
import utils

class Reranker:
    def __init__(self, model_path=None, top_n: int = None):
        if model_path is None:
            model_path = utils.get_reranker_path()
        self.top_n = top_n
        self.reranker = OpenVINOReranker(model_name_or_path=model_path,
                                         model_kwargs={"device": utils.get_device()})
        if top_n is not None:
            self.reranker.top_n = top_n

    def _compute_reranking_score(self, query: str, passages: list[str]):
        class Request(object):
            pass
        request = Request()
        request.query = query
        request.passages = [{"id": i, "text": passage} for i, passage in enumerate(passages)]
        return self.reranker.rerank(request)

    def do_rerank(self, question: str, passage_list: list, passage_getter=None) -> list:
        """
        :param question: received question
        :param passage_list: list of structure that contains passage, representing relevant chunks given by embedding
        :param passage_getter: way to get passage in the structured list, keep None while passing a list of pure passage
        :return: top_n most relevant results in passage_list
        """
        passages = passage_list if passage_getter is None else [passage_getter(element) for element in passage_list]
        scores = self._compute_reranking_score(question, passages)
        result = [passage_list[element["id"]] for element in scores[:self.top_n]]
        return result
