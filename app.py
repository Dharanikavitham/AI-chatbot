import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from gtts import gTTS
import base64
import io
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="AIzaSyBf3UoJMlcLhlmSNSnpIGXXwsBQUCjRQtU")



# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key="AIzaSyBf3UoJMlcLhlmSNSnpIGXXwsBQUCjRQtU",
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context". 
    Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key="AIzaSyBf3UoJMlcLhlmSNSnpIGXXwsBQUCjRQtU",
        model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=False)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response

def text_to_speech(text):
    tts = gTTS(text)
    audio = io.BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    return audio

def main():
    st.set_page_config(
        page_title="Alisa PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Alisa🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                full_response = ''
                for item in response['output_text']:
                    full_response += item

                # Convert Gemini-Pro's response to speech
                audio_data = text_to_speech(full_response)

                # Display the audio using st.audio with autoplay JavaScript
                audio_base64 = base64.b64encode(audio_data.read()).decode("utf-8")
                audio_html = f'<audio src="data:audio/wav;base64,{audio_base64}" autoplay="true" controls style="display:none;"></audio>'
                st.markdown(audio_html, unsafe_allow_html=True)

                # Display the chat message
                st.markdown(f"**Alisa:** {full_response}", unsafe_allow_html=True)

                # Store the response for further use
                st.session_state.response = response

    # Check if the response variable is defined before accessing it
    if "response" in st.session_state:
        message = {"role": "assistant", "content": st.session_state.response}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
