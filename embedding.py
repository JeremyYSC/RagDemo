import os
from chromadb import EmbeddingFunction
from langchain_community.embeddings import OpenVINOBgeEmbeddings
import utils

class EmbeddingWrapper(EmbeddingFunction):
    def __init__(self):
        super().__init__()
        model_path = os.path.join("ov_model", "bge-m3-weight-int4")
        embedding_device = utils.device_widget()
        self.embedding = OpenVINOBgeEmbeddings(
            model_name_or_path=model_path,
            model_kwargs={"device": embedding_device.value, "compile": False},
            encode_kwargs={
                "mean_pooling": False,
                "normalize_embeddings": True,
                "batch_size": 256},
        )
        self.embedding.ov_model.compile()

    def __call__(self, sentences):
        return self.embedding.embed_documents(sentences)

if __name__ == '__main__':
    query_list = ["What is BGE M3?", "Definition of BM25"]
    sentence_list = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    import numpy as np
    embedding_function = EmbeddingWrapper()
    query_embeddings = np.array(embedding_function.__call__(query_list))
    sentence_embeddings = np.array(embedding_function.__call__(sentence_list))
    similarity = query_embeddings @ sentence_embeddings.T
    print(similarity)
    print(query_embeddings.shape)