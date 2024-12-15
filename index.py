import streamlit as st
import subprocess

def main():
    st.title("Chatbots")

    # Add two empty buttons
    button1 = st.button("Alisa-PRO")
    button2 = st.button("Alisa-PDF")

    # Display a message when Button 1 is clicked
    if button1:
        st.write("You want me as a Chatbot")

        # Open another Streamlit interface when Button 1 is clicked
        subprocess.Popen(["streamlit", "run", r"C:\Users\dharani\Desktop\mini project\main.py"])

    # Display a message when Button 2 is clicked
    if button2:
        st.write("You Want me as a PDF Bot")
        subprocess.Popen(["streamlit", "run", r"C:\Users\dharani\Desktop\mini project\app.py"])
if __name__ == "__main__":
    main()
