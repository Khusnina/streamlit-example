import streamlit as st  
import pandas as pd
import numpy as np
import io
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

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
      st.info('Select one either to clean or not.', icon="ℹ️")
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
      df['Description'] = df['Description'].replace('â', '')
      st.write("List of Fiction Book after cleaning")
      st.write(df.head(20))
      stopwords = st.checkbox('Stopwords')
      if stopwords:
         stop_words = list(STOP_WORDS)
         st.write("List of stop words:\n", stop_words[:100])
      contraction = st.checkbox('Contraction Map')
      if contraction:
         st.write("""
            "ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
""")
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
         st.write("filename:", uploaded_file.name)
      content = st.checkbox('Show the content')
      if content:
         st.write(dataframe)
         st.write(dataframe.head(20))
      summarize = st.form_submit_button("Summarize")
   """
   col1,col2 = st.columns(2)
   if summarize:
      st.write(raw_text)
      st.button("Copy text")
      st.write("Words:")
      with col1:
         st.info("Results")
      with col2:
         st.info("Tokens")
   """
 
if choice == 'Statistics':
   st.write("")
