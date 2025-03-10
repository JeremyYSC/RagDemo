from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# List of URLs to load documents from
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm",
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]


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
    if os.path.exists('vector_store.parquet'):
        print("vectorstore exists.")
    else:
        # Initialize a text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
    
        # Split the documents into chunks
        doc_splits = text_splitter.split_documents(docs_list)
    
        # Create embeddings for documents and store them in a vector store
        vectorstore_to_save = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            # > ollama pull bge-m3
            embedding=OllamaEmbeddings(model="bge-m3"),
            persist_path="vector_store.parquet",
            serializer="parquet"
        )
        # Save the vector store to a local file
        vectorstore_to_save.persist()

    # Load the vector store from the local file
    vectorstore_from_load = SKLearnVectorStore(
        embedding=OllamaEmbeddings(model="bge-m3"),
        persist_path="vector_store.parquet",
        serializer="parquet"
    )
    retriever = vectorstore_from_load.as_retriever(k=4)

    # Define the prompt template for the LLM
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    # Initialize the LLM with Llama 3.1 model
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    # Create a chain combining the prompt template and LLM
    rag_chain = prompt | llm | StrOutputParser()

    # Initialize the RAG application
    rag_application = RAGApplication(retriever, rag_chain)
    # Example usage
    question = "What is prompt engineering"
    answer = rag_application.run(question)
    print("Question:", question)
    print("Answer:", answer)


if __name__ == '__main__':
    main()
