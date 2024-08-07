import pymupdf as fitz
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import os
from dotenv import load_dotenv
load_dotenv()
import sys
api_key=os.getenv("OPENAI_API_KEY")
sec_key=os.getenv("huggingfacehub_api_token")


def read_pdf(folder_path):

    detected_text=''
    num_pages=0
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
    for pdf in pdf_files:
        pdf_reader=fitz.open(pdf)
        for page_num in range(pdf_reader.page_count):
            page=pdf_reader.load_page(page_num)
            text=page.get_text("text")
            num_pages+=1
            detected_text+=text+'\n'
        pdf_reader.close()
    return detected_text

def get_vectorstore(file):
    text=read_pdf(file)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."], chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.create_documents([text])
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_index = FAISS.from_documents(text_chunks, embeddings)

    return vector_index


# def create_embeddings_and_save(file, api_key):
#     text=read_pdf(file)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     text_chunks = splitter.create_documents([text])
#     embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-ada-002")
#     vector_index = FAISS.from_documents(text_chunks, embeddings)
#     return vector_index
def trim_chat_history(chat_history):
    return chat_history[-10:]


def create_chain(vectorStore):
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    # model = HuggingFaceEndpoint(
    #     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    #     # repo_id="microsoft/Phi-3-mini-4k-instruct",
    #     task="text-generation",
    #     max_new_tokens=512,
    #     do_sample=False,
    #     timeout=300,
    #     huggingfacehub_api_token=sec_key,
    #     repetition_penalty=1.03,
    # )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

    # contextualize_que_system_prompt = """Based on the provided chat history and the latest user question, which may reference information in the chat history,
    #     your task is to break down the user question into clear, standalone sub-queries that do not depend on the chat history for context.
    #     Ensure you capture all parts of the question, even if there are multiple aspects. Do not answer the question; simply reformulate it into sub-queries if necessary, or leave it unchanged if no reformulation is needed."""
    #
    contextualize_que_system_prompt = """Given the provided chat history and the latest user question, which may reference information from the chat history,
            your task is to break down the user's query into clear, standalone sub-queries that can be understood independently of the chat history.
            Ensure that each part of the query is addressed, even if there are multiple aspects. If any part of the query might be ambiguous or could apply to multiple contexts, include a sub-query asking for clarification from the user.

            Do not provide answers at this stage. Instead, reformulate the user's question into sub-queries if necessary,or retain the original form if no reformulation is necessary.
            If the sub-queries still result in no relevant context being retrieved, indicate that further clarification might be needed and suggest asking the question again with more detail if appropriate.
            Ensure that the chat history is considered to maintain context and continuity in the conversation."""

    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_que_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm=model, retriever=retriever, prompt=retriever_prompt)

    # qa_system_prompt = """You are an assistant specializing in answering questions based strictly on the provided context.\
    #     Your task is to generate a response using only the information within the given context.\
    #     Do not include any external information or assumptions.\
    #     If the answer is not found in the provided context, or If you don't know the answer, just say that you don't know. \
    #     Ensure your answers are only from the provided context, accurate, and maintain a friendly tone.\
    #     <context>
    #     {context}
    #     </context>"""

    # qa_system_prompt = """You are an assistant responsible for answering questions based solely on the information provided in the document context.
    #     Your goal is to thoroughly search the provided context for answers to each sub-query. Use only the information from the uploaded document to formulate your responses.
    #     Do not include any information that is not present in the document.
    #     If the answer to any sub-query is not found in the provided document, kindly reply with 'Sorry, I cannot provide information on that topic based on the current document.'
    #     Please ensure your responses are friendly, helpful, and as informative as possible, based on the context given.
    #     If a query matches multiple contexts (e.g., configuration files for different clouds), ask the user for clarification on which specific context they are referring to.
    #     If the query is clear and there are multiple relevant pieces of information, provide all matched answers.
    #     <context>
    #     {context}
    #     </context>"""

    qa_system_prompt = """You are an assistant responsible for answering questions based solely on the information provided in the document context.
        Your goal is to thoroughly search the provided context for answers to each sub-query. Use only the information from the uploaded document to formulate your responses.
        Do not include any information that is not present in the document.
        Track all parts of the user's question, even if there are multiple questions, and provide comprehensive answers for each.
        If a query matches multiple contexts, ask the user for clarification on which specific context they are referring to, or retrieve and compile all relevant information for each context.
        If the answer to any sub-query is not found in the provided document, kindly reply with 'Sorry, I cannot provide information on that topic based on the current document.
        If there are multiple relevant pieces of information, provide all matched answers and answers in detail with all the steps mentioned.
        Please ensure your responses are friendly, helpful, and as informative as possible, based on the context given.
        After addressing each sub-query, compile the responses into a coherent answer for the user.
        <context>
        {context}
        </context>"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    retrieval_chain = create_retrieval_chain(history_aware_retriever, chain)

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": trim_chat_history(chat_history),
        "input": question,
    })
    return response["answer"]


if __name__ == '__main__':

    file = r"C:\Users\asua\books"
    vectorStore = get_vectorstore(file)
    # Create conversation chain
    chain = create_chain(vectorStore)
    # Initialize chat history
    chat_history = []

    while True:
        user_input = input("\nYou : ")
        if user_input.lower() == "exit":
            break
        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        # print(trim_chat_history(chat_history))
        print("Assistant : ", response)
