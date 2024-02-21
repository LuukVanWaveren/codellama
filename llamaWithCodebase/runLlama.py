
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def loadVectorStore():
    print("loading indexes")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = FAISS.load_local("E:/Work/Projects/Repos/codellama/llamaWithCodebase/vectorDatabases", embeddings)
    retriever = faiss_index.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    print("index loaded")
    return retriever

def loadModel(llm_path):
    callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler])
    return LlamaCpp(model_path=llm_path, n_ctx=2000, callback_manager=callback_manager, verbose=True, use_mlock=True,
                   n_gpu_layers=30, n_threads=4, max_tokens=4000)

def loadModel2(llm_path):
    callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler])
    return LlamaCpp(model_path=llm_path)

def runModel(llm, retriever):
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    chat_history = []
    while (True):
        print("Enter a question: ")
        question = input()
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(result['answer'])

if __name__ == '__main__':
    # llm = loadModel2('E:/Work/Projects/Repos/codellama/CodeLlama-7b/tokenizer.model')
    # runModel(llm,loadVectorStore())

    with open('E:\Work\Projects\Repos\codellama\CodeLlama-7b\params.json', 'r') as file:
        content = file.read()
        print(content)