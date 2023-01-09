import streamlit as st  
import pandas as pd
import numpy as np
import io
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES</h1>", unsafe_allow_html=True)
st.markdown("<hr size='5' width='100%;'>", unsafe_allow_html=True)
activities = ["Introduction","Fiction Books","Summarize","Statistic"]
choice = st.sidebar.selectbox("Select Activity", activities)

if choice == 'Introduction':
   st.write("The extractive text summarization system creates summaries by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary.")
   
if choice == 'Fiction Books':
   st.markdown("<h2 style='text-align: center; color: white;'>FICTIONS BOOKS</h2>", unsafe_allow_html=True)
   category = ["Story","Harry Potter"]
   url = 'https://raw.githubusercontent.com/Khusnina/streamlit-example/master/listBook.csv'
   df = pd.read_csv(url,encoding="latin-1")
   st.write("List of Fiction Book")
   st.write(df.head(20))
   st.write("Shape")
   st.write(df.shape)
   st.write("Info")
   buffer = io.StringIO()
   df.info(buf=buffer)
   s = buffer.getvalue()
   st.text(s)
  
   clean = st.radio("Cleaning the data",('Select', 'Clean', 'Do not clean')) 
   if clean == 'Select':
      st.info('You do not want to clean the list.', icon="ℹ️")
   if clean == 'Clean':
      st.info('You want to clean the list.', icon="ℹ️")
      # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
      df['Description'] = df['Description'].str.replace('\n\n\n\n', ' ')
      df['Description'] = df['Description'].str.replace('\n\n', ' ')
      df['Description'] = df['Description'].str.replace('\n', ' ')
      df['Description'] = df['Description'].str.replace('/', ' ')
      df['Description'] = df['Description'].str.replace('    ', ' ')
      df['Description'] = df['Description'].str.replace('   ', ' ')
      df['Description'] = df['Description'].replace('? ', '. ')
      df['Description'] = df['Description'].replace('*', '')
      df['Description'] = df['Description'].replace('; ', '')
      df['Description'] = df['Description'].replace(', ', '')
      df['Description'] = df['Description'].replace('â ', '')
      st.write("List of Fiction Book after cleaning")
      st.write(df.head(20))
      stopwords = st.checkbox('Stopwords')
      if stopwords:
         st.write("\n['he','she','both', 'hereby', 'because', '’m', 'beyond', 'give', 'becoming', 'so', 'sometime', 'whereupon', 'what', 'none', 'somehow', 'why', 'must', 'nothing', 'when', 'whereby', 'thence', 'well', 'already', 'although', 'formerly', 'anyone', 'around', 'eleven', 'should', 'anything', 'he', 'made', 'bottom', 'with', 'either', 'whether', 'amount', 'five', 'nor', 'a', 'itself', 'thereupon', 'not', 'otherwise', 'per', 'just', 'few', 'these', 'every', 'though', 'herein', 'does', 'quite', 'yourselves', 'afterwards', 'the', 'out', 'part', 'however', 'never', 'was', 'less', 'whatever', 'nine', 'keep', 'third', 'would', '’ll', 'her', 'anywhere', 'move', 'indeed', 'whose', 'upon', 'along', 'call', 'is', 'became', 'down', 'n‘t', 'hers', 'wherever', 'fifteen', 'see', 'top', 'no', 'hence', '’s', 'himself', 'that', 'd', 'two', 'herself', 'seem', 'to', 'there', 'several', 'anyhow', 'seemed', '‘ve', 'elsewhere', 'say', 'others', 'therefore', 'then', 'anyway', 'who', 'unless', 'further', 'noone', 'wherein', 'nobody', 'latter', 'm', 'of', 'behind', 'everyone', 'last', 'due', 'perhaps', 'side', 'thru', 'whither', 'now', 'yourself', 'latterly', 'will', 'against', 'sixty', 'him', 'my', 'four', 'but', 'its', 'most', 'i', 're', 'many', 'about', 'very', 'still', '‘d', 'between', 'forty', 'were', 'beforehand', 'even', 'whenever', 'something', 'or', 'same', 'them', 'besides', 'if', 'have', 'had', 'up', 'back', 'at', 'always', 're', 'above', 'get', 'under', 'for', 'serious', 'themselves', 'his', 'really', 'than', 'thus', 'nevertheless', 'some', 'whom', 'sometimes', 'amongst', 'throughout', 'n’t', 'as', 'regarding', 'they', 'might', 'those', 'are', 'neither', 'whole', 'within', 'yours', 'thereby', 'another', 's', 'former', 'using', 'towards', 'other', 'hereupon', 'from', 'six', 'after', 'me', 'among', 'please', 'whence', 'below', 'may', 'rather', 'somewhere', 'into', '’ve', 'together', 'we', 'ever', 'again', 'various', 'more', 'through', '’d', 'doing', 'our', 'toward', 'us', 'myself', 'front', 'before', 'meanwhile', 'thereafter', 'show', 'often', 'has', 'any', 'empty', 'everything', 'hereafter', 'did', 'their', 'while', 'without', 'over', 'your', 'enough', 'by', 'and', '‘ll', 'on', 'been', 'during', 'name', 'eight', 'make', 'seeming', 'take', 'an', 'ourselves', 'moreover', 'namely', 'become', 'can', 'much', 'she', 'alone', 'seems', 'also', 'beside', 'could', 'ten', 'next', 'whoever', 'hundred', 'put', 'cannot', 'it', 'least', 'first', 'here', 'all', 'twenty', 'whereafter', 'three', 'fifty', 'whereas', 'am', 'own', 'off', 'therein', 'almost', 'm', 'only', 'too', 'this', 'being', 'used', 'twelve', 'across', 'someone', 'full', 'how', 'nowhere', 'one', 'mine', 'each', '‘re', 'done', '‘s', 'in', 'such', 'll', 've', 'onto', 'since', 'do', 'yet', 'except', 'once', 'go', 'everywhere', 'ca', 'where', 'be', 'becomes', 'else', 'mostly', 'ours', 'which', 'until', 'you', '’re', 'via']")
   if clean == 'Do not clean':
      st.info('You do not want to clean the list.', icon="ℹ️")
   option = st.selectbox('Select Category', category)
   if option == 'Story':
      st.write("Select the box to view the content")
      book1 = st.checkbox('Adventures of Huckleberry Finn')
      if book1:
         st.write(df['Title'][0])
         st.write(df['Description'][0])
      book2 = st.checkbox('A Ghost of A Chance')
      if book2:
         st.write(df['Title'][1])
         st.write(df['Description'][1])
      book3 = st.checkbox('Grimm Fairy Tales')
      if book3:
         st.write(df['Title'][2])
         st.write(df['Description'][2])
      book4 = st.checkbox('Anais of Brightshire')
      if book4:
         st.write(df['Title'][3])
         st.write(df['Description'][3])
      book5 = st.checkbox('A Princess of Mars')
      if book5:
         st.write(df['Title'][4])
         st.write(df['Description'][4])
      book6 = st.checkbox('Ardath')
      if book6:
         st.write(df['Title'][5])
         st.write(df['Description'][5])
      book7 = st.checkbox('Heart of Darkness')
      if book7:
         st.write(df['Title'][6])
         st.write(df['Description'][6])
      book8 = st.checkbox('Ella Eris and The Pirates of Redemption')
      if book8:
         st.write(df['Title'][7])
         st.write(df['Description'][7])
            
   if option == 'Harry Potter':
      st.write("Select the box to view the content")
      book9 = st.checkbox('[1]Harry Potter - The Boy Who Lived')
      if book9:
         st.write(df['Title'][8])
         st.write(df['Description'][8])
      book10 = st.checkbox('[2]Harry Potter - The Worst Birthday')
      if book10:
         st.write(df['Title'][9])
         st.write(df['Description'][9])
      book11 = st.checkbox('[3]Harry Potter - Owl Post')
      if book11:
         st.write(df['Title'][10])
         st.write(df['Description'][10])
      book12 = st.checkbox('[4]Harry Potter - The Riddle House')
      if book12:
         st.write(df['Title'][11])
         st.write(df['Description'][11])
      book13 = st.checkbox('[5]Harry Potter - Dudley Demented')
      if book13:
         st.write(df['Title'][12])
         st.write(df['Description'][12])
      book14 = st.checkbox('[6]Harry Potter - The Other Minister')
      if book14:
         st.write(df['Title'][13])
         st.write(df['Description'][13])
      book15 = st.checkbox('[7]Harry Potter - The Dark Lord Ascending')
      if book15:
         st.write(df['Title'][14])
         st.write(df['Description'][14])
   
if choice == 'Summarize':
   st.subheader("EXTRACTIVE TEXT SUMMARIZER")
   with st.form(key = 'nlpForm'):
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
      submit_button = st.form_submit_button(label = 'Summarize')
   col1,col2 = st.columns(2)
   if submit_button:
      with col1:
         st.info("Results")
      with col2:
         st.info("Tokens")
   """
   if st.button("Summarize"):
      st.write(raw_text)
      st.button("Copy text")
      st.write("Words:")
   """
 
if choice == 'Statistics':
   st.write("")
