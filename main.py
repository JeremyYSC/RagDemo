import os
import chromadb
import time

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from rich import print as pprint
from chromadb.api.types import EmbeddingFunction


# 自定義適配器，將 OllamaEmbeddings 包裝為 ChromaDB 相容的嵌入函數
class OllamaEmbeddingWrapper(EmbeddingFunction):
    def __init__(self, model_name: str = "bge-m3"):  # 提供預設值，避免必須參數
        super().__init__()  # 調用基類的 __init__
        self.embedding_model = OllamaEmbeddings(model=model_name)

    def __call__(self, input):  # 符合 ChromaDB 的新介面要求
        return self.embedding_model.embed_documents(input)


# List of URLs to load documents from
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm",
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 初始化 Ollama 模型
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

# 定義生成頁面摘要的 Prompt 模板
page_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    請閱讀以下內容，並生成一份不超過100字的摘要，無法摘要則輸出NULL，不需任何標題，直接輸出內容：

    {text}

    摘要：
    """
)

# 定義回答問題的 Prompt 模板
answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    根據以下內容回答問題，並列出參考文件的檔名與頁數：

    問題：{question}
    內容：
    {context}

    答案：
    """
)

dbpath = "./"  # ChromaDB 儲存路徑
chroma_client = chromadb.PersistentClient(path=dbpath)

collection = chroma_client.get_or_create_collection(
    name="pdf_summaries",
    metadata={"hnsw:space": "cosine"},
    embedding_function=OllamaEmbeddingWrapper()
)


# 函數：處理單個 PDF 文件的每一頁
def process_pdf_pages(file_path):
    try:
        # 使用 PyPDFLoader 讀取 PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 儲存每一頁的摘要
        ids = []
        contents = []
        metadatas = []

        # 遍歷每一頁
        for i, doc in enumerate(documents, 1):
            page_text = doc.page_content

            # 生成該頁的摘要
            rag_chain = page_summary_prompt | llm | StrOutputParser()
            summary = rag_chain.invoke({"text": page_text})

            # page_summaries.append(f"第 {i} 頁摘要:\n{summary}")
            print(f"第 {i} 頁摘要:\n{summary}")

            metadata = {
                "file": file_path,
                "page": i
            }

            ids.append(f"{os.path.basename(file_path)}_page_{i}")
            contents.append(summary)
            metadatas.append(metadata)

        # 將摘要存入 ChromaDB
        collection.add(
            documents=contents,
            ids=ids,
            metadatas=metadatas
        )
    except Exception as e:
        print([f"處理文件 {file_path} 時發生錯誤：{str(e)}"])


# 主函數：遍歷根目錄下所有 PDF 文件並為每頁生成摘要
def summarize_all_pdfs_in_directory(root_path):
    # 確保根目錄存在
    if not os.path.exists(root_path):
        print(f"{root_path} 不存在！")
        return

    # 使用 os.walk 遍歷根目錄及其所有子資料夾
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(dirpath, filename)
                print(f"\n正在處理: {file_path}")
                process_pdf_pages(file_path)


# 函數：從 ChromaDB 查詢並回答問題
def answer_question_from_chroma(question: str, top_k: int = 10):
    start_time = time.time()

    # 從 ChromaDB 查詢最接近的 10 筆摘要
    results = collection.query(
        query_texts=[question],
        n_results=top_k,  # 返回最多 10 個結果
        include=["metadatas", "distances", "documents"]
    )

    print(f"存取DB時間: {time.time() - start_time:.3f} 秒")

    # 提取查詢結果
    summaries = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 儲存原文內容
    context = ""

    # 根據元數據提取原始 PDF 頁面內容
    for summary, metadata, distance in zip(summaries, metadatas, distances):
        file_path = metadata["file"]
        page_num = metadata["page"]

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            original_text = documents[page_num - 1].page_content  # 頁碼從 1 開始，索引從 0 開始
            context += f"\n---\n文件: {file_path}, 第 {page_num} 頁 (距離: {distance:.3f})\n摘要: {summary}\n原文:\n{original_text}\n"
        except Exception as e:
            context += f"\n---\n文件: {file_path}, 第 {page_num} 頁\n錯誤: 無法載入原文 ({str(e)})\n"

    # # 使用 LLM 回答問題
    # print(context)
    print(f"Load文件時間: {time.time() - start_time:.3f} 秒")
    rag_chain = answer_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"question": question, "context": context})
    print(f"回答問題時間: {time.time() - start_time:.3f} 秒")
    return answer


# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


def main():
    # 指定根目錄路徑
    # root_directory = "./"  # 從當前目錄開始遍歷，也可以替換為其他路徑

    # 執行摘要生成
    # summarize_all_pdfs_in_directory(root_directory)

    print(answer_question_from_chroma("請問政府對於環境議題的努力?"))

    # Initialize a text splitter with specified chunk size and overlap
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=10, chunk_overlap=0, separators=['\n']
    # )

    # Split the documents into chunks
    # doc_splits = text_splitter.split_documents(file_context)

    # Create embeddings for documents and store them in a vector store
    # vectorstore = SKLearnVectorStore.from_documents(
    #     documents=doc_splits,
    #     embedding=OllamaEmbeddings(model="nomic-embed-text"),
    # )
    # retriever = vectorstore.as_retriever(k=4)

    # Define the prompt template for the LLM
    # prompt = PromptTemplate(
    #     template="""You are an assistant for question-answering tasks.
    #     Use the following documents to answer the question.
    #     If you don't know the answer, just say that you don't know.
    #     Use three sentences maximum and keep the answer concise:
    #     Question: {question}
    #     Documents: {documents}
    #     Answer:
    #     """,
    #     input_variables=["question", "documents"],
    # )

    # Initialize the LLM with Llama 3.1 model
    # llm = ChatOllama(
    #     model="llama3.1",
    #     temperature=0,
    # )

    # Create a chain combining the prompt template and LLM
    # rag_chain = prompt | llm | StrOutputParser()

    # Initialize the RAG application
    # rag_application = RAGApplication(retriever, rag_chain)
    # Example usage
    # question = "What is prompt engineering"
    # answer = rag_application.run(question)
    # print("Question:", question)
    # print("Answer:", answer)


if __name__ == '__main__':
    main()
