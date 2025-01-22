import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv



from langchain_pinecone import PineconeVectorStore


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # Specify the model
    openai_api_key=OPENAI_API_KEY
)




index_name = "chatbot-memory"

from langchain_pinecone import PineconeVectorStore

# from langchain.document_loaders import PyPDFLoader

# def upload_data(file_path: str, chunk_size=1000, chunk_overlap=200):
    # try:
    #     # Load PDF using PyPDFLoader
    #     loader = PyPDFLoader(file_path)
    #     documents = loader.load()

    #     # Split documents into chunks
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #     documents = text_splitter.split_documents(documents)

    #     try :
    #         vectorstore = PineconeVectorStore.from_documents(
    #             documents=documents,
    #             index_name=index_name,
    #             embedding=embeddings
    #         #     embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY),
    #         )

    #         print(f"Successfully added {len(documents)} documents to Pinecone index '{index_name}'.")
    #     except Exception as e:
    #         print(e)

    # except Exception as e:
    #     print(e)

from langchain_community.document_loaders.csv_loader import CSVLoader

def upload_data(file_path: str, chunk_size=1000, chunk_overlap=200):
    try:
        # Load CSV using CSVLoader
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(documents)

        # Initialize Pinecone vector store
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )

        print(f"Successfully added {len(documents)} documents to Pinecone index '{index_name}'.")
    except Exception as e:
        print(f"Error processing CSV file: {e}")

def get_vectorestore_retriever():
    try:
        print("Loading Pinecone vector store...")
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
            # embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY),
        )
        return vectorstore.as_retriever()
    except Exception as e:
        print(f"Error loading Pinecone vector store: {e}")

        # Load documents

store = {}
# ---: sort store, embeddings
# Create the conversational RAG chain
def create_rag_chain():
    print(store)
    llm_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    retriever = get_vectorestore_retriever()
    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm_model, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        print(store)
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

# Chatbot interaction
def get_response(session_id: str, user_input: str):
    conversational_rag_chain = create_rag_chain()
    return conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )["answer"]
