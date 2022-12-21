import streamlit as st

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.title("ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES\n..............................................................................................................................................................................................")
st.write("The extractive text summarization system creates summaries by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary.")
activities = ["Information","Summarize"]
choice = st.sidebar.selectbox("Select Activity",activities)   

if choice == 'Summarize':
   st.subheader("TEXT SUMMARIZER")
   raw_text = st.text_area("Enter Text Here","Text Content")
   if st.button("Summarize"):
      st.write(raw_text)
   
