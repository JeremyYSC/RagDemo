from FlagEmbedding import BGEM3FlagModel
from chromadb import EmbeddingFunction
import utils

# 自定義適配器，將 OllamaEmbeddings 包裝為 ChromaDB 相容的嵌入函數
class EmbeddingWrapper(EmbeddingFunction):
    def __init__(self):
        super().__init__()  # 調用基類的 __init__
        self.embedding_model =  BGEM3FlagModel(utils.get_embedding_path(), use_fp16=True)

    def __call__(self, sentences):  # 符合 ChromaDB 的新介面要求
        return self.embedding_model.encode(sentences,
                                 batch_size=256,
                                 max_length=512,
                                 return_dense=True,
                                 return_sparse=False,
                                 return_colbert_vecs=False)['dense_vecs']

class EmbeddingModel:
    def __init__(self):
        self.model = BGEM3FlagModel(utils.get_embedding_path(), use_fp16=True)

    def encode(self, sentences):
        """Encode the sentences

        Args:
            sentences (List[str]): The input sentences to encode.

        Returns:
            np.ndarray, shape = (len(sentences), 1024)

        dense_vecs: dense embedding, size=1024
        lexical_weights: lexical matching
        colbert_vecs: Multi-Vector (ColBERT), size=n*1024
        """
        return self.model.encode(sentences,
                                 batch_size=256,
                                 max_length=512,
                                 return_dense=True,
                                 return_sparse=False,
                                 return_colbert_vecs=False)['dense_vecs']

if __name__ == '__main__':
    query_list = ["What is BGE M3?", "Definition of BM25"]
    sentence_list = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embedding_model = EmbeddingModel()
    query_embeddings = embedding_model.encode(query_list)
    sentence_embeddings = embedding_model.encode(sentence_list)

    similarity = query_embeddings @ sentence_embeddings.T
    print(similarity)