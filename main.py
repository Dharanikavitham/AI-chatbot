import webbrowser
import os
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
import base64
import io
from googlesearch import search
import google.generativeai as gen_ai

# Load environment variables
load_dotenv()

# Set up Google Gemini-Pro AI model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key="AIzaSyBf3UoJMlcLhlmSNSnpIGXXwsBQUCjRQtU")
model = gen_ai.GenerativeModel('gemini-pro')

# Function to convert text to speech and return audio data
def text_to_speech(text):
    tts = gTTS(text)
    audio = io.BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)

    return audio

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Alisa-Pro!",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

# Display the chatbot's title on the page
st.title("🤖 Alisa Pro - ChatBot")

# Mute switch
mute_audio = st.checkbox("Mute Text-to-Speech")

# Display the chat history
for message in st.session_state.chat_session.history:
    st.markdown(message.parts[0].text)

# Input field for the user's message
user_prompt = st.chat_input("Ask Alisa-Pro...")
if user_prompt:
    # Add the user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    if "open youtube" in user_prompt.lower():
        st.chat_message("assistant").markdown("Sure, opening YouTube for you.")
        # Open YouTube in the default web browser
        webbrowser.open("https://www.youtube.com")
    else:
        # Send the user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

            # Convert Gemini-Pro's response to speech only if not muted
            if not mute_audio:
                audio_data = text_to_speech(gemini_response.text)

                # Convert audio to base64
                base64_audio = base64.b64encode(audio_data.read()).decode("utf-8")

                # Rewind the audio_data
                audio_data.seek(0)

                # Autoplay audio using JavaScript
                st.markdown(f'<audio autoplay src="data:audio/wav;base64,{base64_audio}"></audio>', unsafe_allow_html=True)
