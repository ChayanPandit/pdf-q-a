import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain



import asyncio

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 1000,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
    
def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    
def get_conversation_chain():
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":512})
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain



def get_conversation_chain_gemini():

    prompt_template = """
    The foloowing question can be a standalone question or a follow up question based on the chat history.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Chat History:\n{chat_history}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context","chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def handle_user_input(user_input):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input)

    chain = get_conversation_chain_gemini()
    st.session_state.conversation = chain

    
    response = chain(
        {"input_documents":docs, "chat_history":st.session_state.chat_history, "question": user_input}
        , return_only_outputs=False
        )
    
    # print(response)
    st.write("Reply: ", response["output_text"])
    
    st.session_state.chat_history.append({'question':user_input,'answer':response['output_text']})
    # print(st.session_state.chat_history)
    
    for i, message in enumerate(st.session_state.chat_history):
        # st.write(message['question'],unsafe_allow_html=True)
        st.write(user_template.replace("{{MSG}}",message['question']),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",message['answer']),unsafe_allow_html=True)
    

def main():
    load_dotenv()
    st.set_page_config(page_title="Q&A on PDFs",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    st.header("Q&A on PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    
    if user_question:
        handle_user_input(user_question)
    else:
        st.write(user_template.replace("{{MSG}}","hello bot"), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}","hello human"), unsafe_allow_html=True)
    
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload ypur PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")
    

if __name__ == '__main__':
    main()