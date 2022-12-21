import streamlit as st

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.title("ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES\n..................................................................................................................................................................................................................")
st.title("TEXT SUMMARIZER")
st.write("The extractive text summarization system creates summaries by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary.")
st.title("TEXT SUMMARIZER")
activities = ["Summarize"]
choice = st.sidebar.selectbox("Select Activity",activities)   
with st.form("my_form"):
   st.write("Upload TXT File")
   title = st.text_input('Fiction Book Title')
   st.write('Fiction Books Content', title)
   st.form_submit_button("Summarize")

   
