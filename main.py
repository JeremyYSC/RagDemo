import os
import chromadb
import time

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from chromadb.api.types import EmbeddingFunction
from embedding import EmbeddingWrapper
from reranker import Reranker


# 自定義適配器，將 OllamaEmbeddings 包裝為 ChromaDB 相容的嵌入函數
class OllamaEmbeddingWrapper(EmbeddingFunction):
    def __init__(self, model_name: str = "bge-m3"):  # 提供預設值，避免必須參數
        super().__init__()  # 調用基類的 __init__
        self.embedding_model = OllamaEmbeddings(model=model_name)

    def __call__(self, input):  # 符合 ChromaDB 的新介面要求
        return self.embedding_model.embed_documents(input)


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
    可以選擇是否要參考以下文件回答問題，如果有參考以下內容，請列出參考文件的檔名與頁數：

    問題：{question}
    內容：
    {context}

    答案：
    """
)

dbpath = "./"  # ChromaDB 儲存路徑
chroma_client = chromadb.PersistentClient(path=dbpath)

# 函數：處理單個 PDF 文件的每一頁
def process_pdf_pages(collection, file_path):
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
                "page": i,
                "type": "file"
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

def process_images(vision_language, collection, file_path):
    try:
        summary = vision_language.image_request(image_path=file_path)
        print(f"摘要:\n {summary}")
        collection.add(
            documents=summary,
            ids=[f"{os.path.basename(file_path)}"],
            metadatas=[{
                "file": file_path,
                "page": 0,
                "type": "image"
            }]
        )
    except Exception as e:
        print([f"處理文件 {file_path} 時發生錯誤：{str(e)}"])

# 主函數：遍歷根目錄下所有 PDF 文件並為每頁生成摘要，遍歷根目錄下所有 image 並為每張圖片生成摘要
def summarize_all_files_in_directory(collection, root_path):
    # 確保根目錄存在
    if not os.path.exists(root_path):
        print(f"{root_path} 不存在！")
        return

    # 使用 os.walk 遍歷根目錄及其所有子資料夾
    from vision_language import VisionLanguage
    vision_language = VisionLanguage()
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(dirpath, filename)
                print(f"\n正在處理: {file_path}")
                process_pdf_pages(collection, file_path)
            elif (filename.lower().endswith((".jpg", ".jpeg", ".png")) and
                    "model" not in dirpath.lower() and
                    "venv" not in dirpath.lower()):
                file_path = os.path.join(dirpath, filename)
                print(f"\n正在處理: {file_path}")
                process_images(vision_language, collection, file_path)


# 函數：從 ChromaDB 查詢並回答問題
def answer_question_from_chroma(collection, question: str, top_k: int = 10):
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

    ranker = Reranker(top_n=3)
    reranked_summaries, reranked_metadatas, reranked_distances = ranker.do_rerank_results(question, zip(summaries, metadatas, distances))

    # 儲存原文內容
    context = ""

    # 根據元數據提取原始 PDF 頁面內容
    load_pdf_start_time = time.time()

    for summary, metadata, distance in zip(reranked_summaries, reranked_metadatas, reranked_distances):
        file_path = metadata["file"]
        page_num = metadata["page"]
        data_type = metadata["type"]

        if "file" == data_type:
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                original_text = documents[page_num - 1].page_content  # 頁碼從 1 開始，索引從 0 開始
                context += f"\n---\n文件: {file_path}, 第 {page_num} 頁 (距離: {distance:.3f})\n摘要: {summary}\n原文:\n{original_text}\n"
            except Exception as e:
                context += f"\n---\n文件: {file_path}, 第 {page_num} 頁\n錯誤: 無法載入原文 ({str(e)})\n"
        elif "image" == data_type:
            context += f"\n---\n圖片: {file_path} (距離: {distance:.3f})\n摘要: {summary}\n"
    # # 使用 LLM 回答問題
    # print(context)
    print(f"Load文件時間: {time.time() - load_pdf_start_time:.3f} 秒")
    llm_start_time = time.time()
    rag_chain = answer_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"question": question, "context": context})
    print(f"LLM回答問題時間: {time.time() - llm_start_time:.3f} 秒")
    print(f"總耗時: {time.time() - start_time:.3f} 秒")
    return answer


# 函數：持續提問模式
def interactive_question_mode(collection):
    print("\n進入問答模式，請輸入問題（輸入 'exit' 退出）：")
    while True:
        question = input("問題：")
        if question.lower() == "exit":
            print("退出問答模式")
            break
        if not question.strip():
            print("請輸入有效的問題！")
            continue

        answer= answer_question_from_chroma(collection, question)

        print("\n問題：", question)
        print("\n回答：")
        print(answer)
        print("\n" + "=" * 50)


def main():
    collection = chroma_client.get_or_create_collection(
        name="pdf_summaries",
        metadata={"hnsw:space": "cosine"},
        embedding_function=EmbeddingWrapper()
    )

    # 指定根目錄路徑
    #root_directory = "./"  # 從當前目錄開始遍歷，也可以替換為其他路徑

    # 執行摘要生成
    #summarize_all_files_in_directory(collection, root_directory)

    interactive_question_mode(collection)
    # Q: 俄烏戰爭影響燃料價格，日本政府補貼幾億元以減輕用戶負擔 A:5500億
    # Q: 給我黃色星星的圖片
    # Q: 給我黃色太陽的圖片

if __name__ == '__main__':
    main()
