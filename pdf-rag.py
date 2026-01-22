## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "./data/BOI.pdf"
model = "mitra:latest"

# Local PDF file uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    documentData = loader.load()
    print("done loading....")
else:
    print("Upload a PDF file")

# Preview first page
# content = documentData[0].page_content
# print(content[:100])

# ==== End of PDF Ingestion ====


# ==== Extract Text from PDF Files and Split into Small Chunks ====

from langchain_ollama import OllamaEmbeddings
embeddings=OllamaEmbeddings(model="nomic-embed-text")
print("embeddings generated")

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
documentDataChunks = text_splitter.split_documents(documentData)
print(f"done splitting...., length : {len(documentDataChunks)}")

print(f"Number of documentDataChunks: {len(documentDataChunks)}")
# print(f"Example chunk: {documentDataChunks[0]}")

# ===== Add to vector database ===
from langchain_chroma import Chroma

vector_db = Chroma(
    collection_name="simple_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
document_ids = vector_db.add_documents(documents=documentDataChunks)

print("done adding to vector database....")


## === Retrieval ===
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up our model to use
llm = ChatOllama(model=model)

# a simple technique to generate multiple questions from a single question and then retrieve documents
# based on those questions, getting the best of both worlds.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

from langchain.tools import tool
retrieved_docs = vector_db.similarity_search(query, k=2)
serialized = "\n\n".join(
    (f"Source: {doc.metadata}\nContent: {doc.page_content}")
    for doc in retrieved_docs
)

# # RAG prompt
# template = """Answer the question based ONLY on the following context:
# {context}
# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)


# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# # res = chain.invoke(input=("what is the document about?",))
# # res = chain.invoke(
# #     input=("what are the main points as a business owner I should be aware of?",)
# # )
# res = chain.invoke(input=("how to report BOI?",))

# print(res)
