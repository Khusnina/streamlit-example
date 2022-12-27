import streamlit as st  

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.title("<center> ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES </center>\n..............................................................................................................................................................................................")
activities = ["Introduction","Fiction Books","Summarize","Statistic"]
choice = st.sidebar.selectbox("Select Activity",activities)   

if choice == 'Introduction':
   st.write("The extractive text summarization system creates summaries by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary.")

if choice == 'Fiction Books':
   st.write("")
   
if choice == 'Summarize':
   st.subheader("EXTRACTIVE TEXT SUMMARIZER")
   st.button("Upload File (txt)")
   agree = st.checkbox('Show sentence')
   raw_text = st.text_area("Original Content","Enter text here")
   if st.button("Summarize"):
      st.write("Enter text here")
      st.write(raw_text)
      st.button("Copy text")
      st.write("Words:")
      
   if agree:
    st.write('---')
 
if choice == 'Statistics':
   st.write("")
