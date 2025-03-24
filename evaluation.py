import asyncio
import numpy as np
import time
import utils

from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import QueryBundle
from llama_index.llms.ollama import Ollama
from typing import Any, List


class LocalBGEM3Embedding(BaseEmbedding):
    _model: BGEM3FlagModel = PrivateAttr()
    _use_fp16: bool = PrivateAttr()

    def __init__(
            self,
            model_path: str = None,
            use_fp16: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = BGEM3FlagModel(model_path or utils.get_embedding_path(), use_fp16=use_fp16)
        self._use_fp16 = use_fp16

    @classmethod
    def class_name(cls) -> str:
        return "bge_m3"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        result = self._model.encode([query])
        # 提取 dense_vecs 作為嵌入向量
        embedding = result['dense_vecs'][0]
        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        result = self._model.encode([text])
        # 提取 dense_vecs 作為嵌入向量
        embedding = result['dense_vecs'][0]
        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        result = self._model.encode(texts)
        # 提取 dense_vecs 作為嵌入向量
        embeddings = result['dense_vecs']
        return [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]


class LocalOpenVINOBgeEmbedding(BaseEmbedding):
    """使用 OpenVINOBgeEmbeddings 的本地嵌入模型"""
    _model: OpenVINOBgeEmbeddings = PrivateAttr()

    def __init__(
            self,
            model_path: str = None,
            **kwargs: Any,
    ) -> None:
        """初始化嵌入模型

        Args:
            model_path (str, optional): OpenVINO BGE 模型的路徑
            **kwargs: 其他參數，傳遞給父類
        """
        super().__init__(**kwargs)
        # 初始化 OpenVINOBgeEmbeddings 模型
        self._model = OpenVINOBgeEmbeddings(
            model_name_or_path=model_path,
            model_kwargs={"device": utils.get_device(), "compile": False},
        )
        self._model.ov_model.compile()

    @classmethod
    def class_name(cls) -> str:
        """返回類別名稱"""
        return "openvino_bge"

    def _get_query_embedding(self, query: str) -> List[float]:
        """生成查詢的嵌入向量

        Args:
            query (str): 查詢文本

        Returns:
            List[float]: 查詢的嵌入向量
        """
        embedding = self._model.embed_query(query)
        return embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        """生成單一文本的嵌入向量

        Args:
            text (str): 輸入文本

        Returns:
            List[float]: 文本的嵌入向量
        """
        embedding = self._model.embed_documents([text])[0]
        return embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成多個文本的嵌入向量

        Args:
            texts (List[str]): 輸入文本列表

        Returns:
            List[List[float]]: 多個文本的嵌入向量列表
        """
        embeddings = self._model.embed_documents(texts)
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """異步生成查詢的嵌入向量

        Args:
            query (str): 查詢文本

        Returns:
            List[float]: 查詢的嵌入向量
        """
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """異步生成單一文本的嵌入向量

        Args:
            text (str): 輸入文本

        Returns:
            List[float]: 文本的嵌入向量
        """
        return self._get_text_embedding(text)


class FlagRerankerPostprocessor:
    def __init__(self, reranker, top_k: int = 3):
        self.reranker = reranker
        self.top_k = top_k  # 設定 top_k

    def postprocess_nodes(self, nodes, query_bundle):
        # 將查詢與每個節點的內容組成對，計算相關性分數
        pairs = [(str(query_bundle.query_str), node.node.text) for node in nodes]
        scores = self.reranker.compute_score(pairs)

        # 更新節點的分數並排序
        for node, score in zip(nodes, scores):
            node.score = score

        # 返回 top_k 個節點
        return sorted(nodes, key=lambda x: x.score or 0, reverse=True)[:self.top_k]


class OpenVINORerankerPostprocessor:
    def __init__(self, reranker, top_k: int = 3):
        self.reranker = reranker
        self.top_k = top_k  # 設定 top_k

    def postprocess_nodes(self, nodes, query_bundle):
        class Request:
            pass

        request = Request()
        request.query = str(query_bundle.query_str)
        request.passages = [{"id": i, "text": node.node.text} for i, node in enumerate(nodes)]
        scores = self.reranker.rerank(request)
        result = [nodes[element["id"]] for element in scores[:self.top_k]]
        # print(scores)
        # print(nodes)
        # print(result)

        # 返回 top_k 個節點
        return result


class CustomRetriever(BaseRetriever):
    def __init__(self, retriever, postprocessor):
        self.retriever = retriever
        self.postprocessor = postprocessor

    def _retrieve(self, query: str):
        # 同步檢索並應用後處理
        nodes = self.retriever._retrieve(query)
        query_bundle = QueryBundle(query_str=query)

        if self.postprocessor:
            nodes = self.postprocessor.postprocess_nodes(nodes, query_bundle)
        else:
            nodes = nodes[:5]

        return nodes


def main():
    start_time = time.time()
    documents = SimpleDirectoryReader(input_files=['立法院第11屆第3會期行政院施政報告.pdf']).load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512, separator='\n')
    nodes = node_parser.get_nodes_from_documents(documents[4:9])
    print("node len: " + str(len(nodes)))

    llm = Ollama(
        model="llama3.1:latest",
        request_timeout=120.0,
        temperature=0.0)

    # Prompt to generate questions
    qa_generate_prompt_tmpl = """
    上下文資訊如下。

    ---------------------
    {context_str}
    ---------------------

    鑑於上下文資訊而不是先前知識，僅根據以下查詢產生問題。
    您是一位教授。您的任務是為即將進行的測驗/考試設定 {num_questions_per_chunk} 個問題。
    整份文件中的問題性質應多種多樣。
    問題不應包含選項，不應以 Q1/Q2 開頭。
    將問題限制在所提供的上下文資訊內。
    """
    qa_prompt = PromptTemplate(qa_generate_prompt_tmpl)

    queries = []
    reference_answers = []
    expected_ids = []
    num_questions_per_chunk = 2

    for node in nodes:
        context_str = node.text
        prompt_input = qa_prompt.format(context_str=context_str, num_questions_per_chunk=num_questions_per_chunk)
        response = llm.complete(prompt_input)
        # print(response)

        # 解析回應（假設回應格式為問題和答案逐行列出）
        lines = response.text.split("\n")

        for i in range(0, len(lines), 2):
            if i + 1 < len(lines) and lines[i].strip() and lines[i + 1].strip():
                question_1 = lines[i][2:].strip()
                question_2 = lines[i + 1][2:].strip()

                print(f"問題1: {question_1}")
                print(f"問題2: {question_2}\n")

                queries.append(question_1)
                reference_answers.append(node.text)
                expected_ids.append([node.node_id])
                queries.append(question_2)
                reference_answers.append(node.text)
                expected_ids.append([node.node_id])

    print("queries len: " + str(len(queries)))
    # print(queries)
    print("reference_answers len: " + str(len(reference_answers)))
    # print(reference_answers)
    print("expected_ids len: " + str(len(expected_ids)))

    # embedding_model = LocalBGEM3Embedding(model_path=utils.get_embedding_path(), use_fp16=True)

    embedding_model = LocalOpenVINOBgeEmbedding(model_path=utils.get_openvino_embedding_path())

    index = VectorStoreIndex(nodes=nodes, embed_model=embedding_model)
    retriever = index.as_retriever(similarity_top_k=10)

    # reranker = FlagReranker(model_name_or_path=utils.get_reranker_path(), use_fp16=True, local_files_only=True)
    # postprocessor = FlagRerankerPostprocessor(reranker, top_k=5)

    reranker = OpenVINOReranker(
        model_name_or_path=utils.get_openvino_reranker_path(),
        model_kwargs={"device": utils.get_device()}
    )
    postprocessor = OpenVINORerankerPostprocessor(reranker, top_k=5)

    custom_retriever = CustomRetriever(retriever, postprocessor)

    evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"],
        retriever=custom_retriever
    )

    eval_dataset = [
        {"query": q, "reference_answer": r, "expected_ids": e}
        for q, r, e in zip(queries, reference_answers, expected_ids)
    ]

    async def run_evaluation():
        eval_results = []
        mrr_scores = []
        hit_rate_scores = []

        for item in eval_dataset:
            result = await evaluator.aevaluate(
                query=item["query"],
                reference=item["reference_answer"],
                expected_ids=item["expected_ids"]
            )

            eval_results.append(result)

        for result in eval_results:
            print(f"Query: {result.query}")
            print(f"Retrieved: {result.retrieved_ids}")
            print(f"Expected: {result.expected_ids}")
            print(f"Metrics: {result.metric_dict}")
            mrr_scores.append(result.metric_dict['mrr'].score)
            hit_rate_scores.append(result.metric_dict['hit_rate'].score)

        # 計算平均值
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
        avg_hit_rate = sum(hit_rate_scores) / len(hit_rate_scores) if hit_rate_scores else 0
        print("Sum of mmr: " + str(sum(mrr_scores)))
        print("Sum of hit_rate: " + str(sum(hit_rate_scores)))
        print("avg_mrr: " + str(avg_mrr))
        print("avg_hit_rate: " + str(avg_hit_rate))

    asyncio.run(run_evaluation())
    print(f"總耗時: {time.time() - start_time:.3f} 秒")


if __name__ == '__main__':
    main()
