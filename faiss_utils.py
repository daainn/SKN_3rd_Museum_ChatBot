
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_faiss_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever