import streamlit as st  
import pandas as pd
import numpy as np

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES</h2>", unsafe_allow_html=True)
st.markdown("<hr size='5' width='100%;'>", unsafe_allow_html=True)
activities = ["Introduction","Fiction Books","Summarize","Statistic"]
choice = st.sidebar.selectbox("Select Activity",activities)   

if choice == 'Introduction':
   st.write("The extractive text summarization system creates summaries by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary.")

if choice == 'Fiction Books':
   st.write("")
   df = pd.DataFrame(
   np.random.randn(15, 2),
   columns=('col %d' % i for i in range(2)))
   st.table(df)
   
if choice == 'Summarize':
   st.subheader("EXTRACTIVE TEXT SUMMARIZER")
   st.button("Upload File (txt)")
   agree = st.checkbox('Show sentence')
   raw_text = st.text_area("Original Content","Enter text here")
   uploaded_file = st.file_uploader("Choose a file")
   if uploaded_file is not None:
      # To read file as bytes:
      bytes_data = uploaded_file.getvalue()
      st.write(bytes_data)
      # To convert to a string based IO:
      stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
      st.write(stringio)
      # To read file as string:
      string_data = stringio.read()
      st.write(string_data)
      # Can be used wherever a "file-like" object is accepted:
      dataframe = pd.read_csv(uploaded_file)
      st.write(dataframe)
   if st.button("Summarize"):
      st.write("Enter text here")
      st.write(raw_text)
      st.button("Copy text")
      st.write("Words:")
      
   if agree:
    st.write('---')
 
if choice == 'Statistics':
   st.write("")
