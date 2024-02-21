import os
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def textToChunks(root_dir):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    return texts

def createVectorDatabaseFromChunks(textChunks, path):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = FAISS.from_documents(textChunks, embeddings)
    faiss_index.save_local(path)

def createVectorDatabase(source, destination):
    textChunks=textToChunks(source)
    createVectorDatabaseFromChunks(textChunks, destination)
    print('vector database created')

if __name__ == '__main__':
    createVectorDatabase('E:/Work/Projects/Repos/Python/HHS_MachineLearning_ImageClassifier', 'E:/Work/Projects/Repos/codellama/llamaWithCodebase/vectorDatabases')