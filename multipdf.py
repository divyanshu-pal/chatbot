import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# llm model communicate framwork


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyBtVqa3PmTdk71awDwTeDFamb33TAgyTwU",)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   google_api_key="AIzaSyBtVqa3PmTdk71awDwTeDFamb33TAgyTwU",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",
                                                   google_api_key="AIzaSyBtVqa3PmTdk71awDwTeDFamb33TAgyTwU"
                                                  )    
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    
        chain = get_conversational_chain()   
        # llm model load
        
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)
    
        print("Response::",response)
       
        return response["output_text"]
    except Exception as e :
        print("error: ",e)
        return "Sorry ! not able to generate"
      
    
    


def main():
   
    
    st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout='wide')
    st.title("Chat with Divyanshu&Sanket 📚")

    with st.sidebar:
        st.header("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files here:", type=['pdf'], accept_multiple_files=True)
        if st.button("Process PDF Files"):
            if pdf_docs:
                with st.spinner("Extracting text from PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! Ask any question.")

    # Initialize or update chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ""
    if 'input_counter' not in st.session_state:
        st.session_state.input_counter = 0  # Counter to ensure unique keys

    # Increment key counter each time the page is reloaded to avoid key conflict
    key = f"question_{st.session_state.input_counter}"
    user_question = st.text_input("Ask a Question from the PDF Files", "", key=key)

    if st.button("Submit"):
        if user_question:
            try:
                response = user_input(user_question)  # Assume this function generates a response
                new_entry = f"User: {user_question}\nBot: {response}\n\n"
                st.session_state.chat_history += new_entry
                st.session_state.input_counter += 1  # Update counter after submission
                st.text_input("Ask a Question from the PDF Files", value="", key=f"question_{st.session_state.input_counter}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    if st.session_state.chat_history:
        st.text_area("Chat History", value=st.session_state.chat_history, height=300, disabled=True)

if __name__ == "__main__":
    main()