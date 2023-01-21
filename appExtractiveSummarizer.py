import streamlit as st  
import pandas as pd
import numpy as np
import io
import unidecode
import unicodedata
import re
import time 
import string
import contractions
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from PIL import Image
from string import punctuation
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from nltk.cluster.util import cosine_distance
import networkx as nx
import neattext.functions as nfx

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES</h1>", unsafe_allow_html=True)
st.markdown("<hr size='5' width='100%;'>", unsafe_allow_html=True)
activities = ["💡 Introduction","📚 Fiction Books","📝 Summarize","📊 Statistic"]
choice = st.sidebar.selectbox("💻 Select Activity", activities)

if choice == '💡 Introduction':
   st.markdown("<h2 style='text-align: center; color: white;'>💡 INTRODUCTION 💡</h2>", unsafe_allow_html=True)
   st.markdown("<p style='text-align: center; color: white;'>Books are becoming increasingly popular among readers who love to read books especially English books. The passage of time has changed the source of books that can be read online in the form of electronic format where users only need to use a mobile device via the Internet. Books consist of various types of genres that are categorized into two namely fiction and non-fiction. Fiction books refer to literary plots, background, and characters designed and created from the writer’s imagination. Based on term of book sales in Malaysia, fiction books are more popular among readers than non-fiction book. Readers usually make an online book review after reading a whole book that contains a long story divided into several chapters that give structure and readability to the book by summarizing it manually to describe the contents.</p>", unsafe_allow_html=True)
   image = Image.open('summarization.png')
   col1, col2, col3 = st.columns([12,20,10])
   with col1:
      st.write("")
   with col2:
      st.image(image, caption = 'Text Summarization')
   with col3:
      st.write("")
   st.markdown("<p style='text-align: center; color: white;'>\nThe extractive text summarization system creates summary by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary. The extraction summary method is the selection of sentences or phrases that have the highest score from the original text and organize them into a new, shorter text without changing the source text.</p>", unsafe_allow_html=True)
   st.markdown("<p style='text-align: center; color: white;'>\nFor extractive text summarization, the main sentence and the object are extracted without modifying the object itself, the strategy of concatenating on extracting summary from a given corpus. The system will select the most important sentences in the online book then combine them to form a summary extractively through machine learning with Natural Language Processing (NLP). The Repetitive Neural Network (RNN) is the model that will be used to generate the summary. The extracted summaries were used to train the extractive Recurrent Neural Network (RNN) model that had been used for the classification task and it generated the latest results. Recurrent Neural Network (RNN) is the most common architectures for Neural Text Summarization. Recurrent Neural Network (RNN) have been used widely in text summarization problem. The approach is based on the idea of identifying a set of sentences which collectively give the highest ROUGE with respect to the gold summary.</p>", unsafe_allow_html=True)
    
if choice == '📚 Fiction Books':
   st.markdown("<h2 style='text-align: center; color: white;'>📚 FICTIONS BOOKS 📚</h2>", unsafe_allow_html=True)
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
  
   clean = st.radio("Cleaning the data",('Select', 'Process', 'No Process')) 
   if clean == 'Select':
      st.info('Select one either to process or not.', icon="ℹ️")
   if clean == 'Process':
      st.info('You want to process the list.', icon="ℹ️")
      # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
      df['Description'] = df['Description'].str.replace('\n\n\n\n', ' ')
      df['Description'] = df['Description'].str.replace('\n\n', ' ')
      df['Description'] = df['Description'].str.replace('\n', ' ')
      df['Description'] = df['Description'].str.replace('\r', '')
      df['Description'] = df['Description'].str.replace('/', ' ')
      df['Description'] = df['Description'].str.replace('    ', ' ')
      df['Description'] = df['Description'].str.replace('   ', ' ')
      df['Description'] = df['Description'].replace('? ', '. ')
      df['Description'] = df['Description'].replace('*', '')
      df['Description'] = df['Description'].replace('; ', '')
      df['Description'] = df['Description'].replace(', ', '')
      df['Description'] = df['Description'].replace('â', '')
      st.write("List of Fiction Book after processing")
      st.write(df.head(20))
      st.download_button("Download CSV",
                         df.to_csv(),
                         file_name = 'listBook.csv',
                         mime = 'text/csv')
      stopwords = st.checkbox('Stopwords')
      if stopwords:
         stopwords = nltk.corpus.stopwords.words('english')
         st.write(stopwords[:100])
      contraction = st.checkbox('Contraction Map')
      if contraction:
         st.write("")
   if clean == 'No Process':
      st.info('You do not want to process the list.', icon="ℹ️")
   option = st.selectbox('Select Category', category)
   if option == 'Story':
      st.write("Select the box to view the content")
      book1 = st.checkbox('Adventures of Huckleberry Finn')
      if book1:
         st.write(df['Title'][0])
         st.write(df['Description'][0])
         content1 = df['Description'][0]
         st.download_button('Download', content1)
      book2 = st.checkbox('A Ghost of A Chance')
      if book2:
         st.write(df['Title'][1])
         st.write(df['Description'][1])
         content2 = df['Description'][1]
         st.download_button('Download', content2)
      book3 = st.checkbox('Grimm Fairy Tales')
      if book3:
         st.write(df['Title'][2])
         st.write(df['Description'][2])
         content3 = df['Description'][2]
         st.download_button('Download', content3)
      book4 = st.checkbox('Anais of Brightshire')
      if book4:
         st.write(df['Title'][3])
         st.write(df['Description'][3])
         content4 = df['Description'][3]
         st.download_button('Download', content4)
      book5 = st.checkbox('A Princess of Mars')
      if book5:
         st.write(df['Title'][4])
         st.write(df['Description'][4])
         content5 = df['Description'][4]
         st.download_button('Download', content5)
      book6 = st.checkbox('Ardath')
      if book6:
         st.write(df['Title'][5])
         st.write(df['Description'][5])
         content6 = df['Description'][5]
         st.download_button('Download', content6)
      book7 = st.checkbox('Heart of Darkness')
      if book7:
         st.write(df['Title'][6])
         st.write(df['Description'][6])
         content7 = df['Description'][6]
         st.download_button('Download', content7)
      book8 = st.checkbox('Ella Eris and The Pirates of Redemption')
      if book8:
         st.write(df['Title'][7])
         st.write(df['Description'][7])
         content8 = df['Description'][7]
         st.download_button('Download', content8)
            
   if option == 'Harry Potter':
      st.write("Select the box to view the content")
      book9 = st.checkbox('[1]Harry Potter - The Boy Who Lived')
      if book9:
         st.write(df['Title'][8])
         st.write(df['Description'][8])
         content9 = df['Description'][8]
         st.download_button('Download', content9)
      book10 = st.checkbox('[2]Harry Potter - The Worst Birthday')
      if book10:
         st.write(df['Title'][9])
         st.write(df['Description'][9])
         content10 = df['Description'][9]
         st.download_button('Download', content10)
      book11 = st.checkbox('[3]Harry Potter - Owl Post')
      if book11:
         st.write(df['Title'][10])
         st.write(df['Description'][10])
         content11 = df['Description'][10]
         st.download_button('Download', content11)
      book12 = st.checkbox('[4]Harry Potter - The Riddle House')
      if book12:
         st.write(df['Title'][11])
         st.write(df['Description'][11])
         content12 = df['Description'][11]
         st.download_button('Download', content12)
      book13 = st.checkbox('[5]Harry Potter - Dudley Demented')
      if book13:
         st.write(df['Title'][12])
         st.write(df['Description'][12])
         content13 = df['Description'][12]
         st.download_button('Download', content13)
      book14 = st.checkbox('[6]Harry Potter - The Other Minister')
      if book14:
         st.write(df['Title'][13])
         st.write(df['Description'][13])
         content14 = df['Description'][13]
         st.download_button('Download', content14)
      book15 = st.checkbox('[7]Harry Potter - The Dark Lord Ascending')
      if book15:
         st.write(df['Title'][14])
         st.write(df['Description'][14])
         content15 = df['Description'][14]
         st.download_button('Download', content15)

if choice == '📝 Summarize':
   st.markdown("<h2 style='text-align: center; color: white;'>📝 EXTRACTIVE TEXT SUMMARIZER 📝</h2>", unsafe_allow_html=True)
   with st.form(key = 'nlpForm'):
      text = st.text_area("Original Content","Enter text here")
      submitted = st.form_submit_button("Summarize")
      if submitted:
         st.info("Result")
         st.write(text)
         stopwords = list(STOP_WORDS)
         stopwords
         nlp = spacy.load('en_core_web_sm')
         doc = nlp(text)
         tokens = [token.text for token in doc]
         st.write(doc)
         punctuation+='/n'
         punctuation
         word_freq={}
         for word in doc:
            if word.text.lower() not in stopwords:
               if word.text.lower() not in punctuation:
                  if word.text not in word_freq.keys():
                     word_freq[word.text]=1
                  else:
                     word_freq[word.text]+=1
            st.write(word_freq)
     
      
   uploaded_txt = st.file_uploader("Choose a file",type=["txt"])
   if uploaded_txt is not None:
      st.write(type(uploaded_txt))
      file_details_txt = {"filename":uploaded_txt.name,"filetype":uploaded_txt.type,"filesize":uploaded_txt.size}
      st.write(file_details_txt)
      if uploaded_txt.type =="text/plain":
         Dftxt = uploaded_txt.read()
         raw_text = str(Dftxt,"utf-8")
         st.text(raw_text)
      if st.button('Summarize file'):
         st.success('Stopwords', icon="✅")
         stopWords = list(stopwords.words("english"))+list(punctuation)+list([0,1,2,3,4,5,6,7,8,9])
         stopWords[15:25]
         
         st.success('Word Tokenize', icon="✅")
         words = word_tokenize(raw_text)
         words[15:25]
         
         st.success('Frequent Table', icon="✅")
         freqTable={}
         for word in words: 
            word = word.lower() 
            if word not in stopWords:
               if word in freqTable: 
                  freqTable[word] += 1
               else: 
                  freqTable[word] = 1
         st.write(freqTable)
         st.write("Items")
         st.write(freqTable.items())
         st.write(sorted(freqTable.items(), key = lambda x: x[1]))
         
         st.success('Sent Tokenize', icon="✅")
         sentences = sent_tokenize(raw_text) 
         for sen in sentences:
            st.write(sen,"\n")
                  
         st.success('Sentence Weight', icon="✅")
         sentence_weight = dict() 
         for sentence in sentences: 
            # print(sentence,'\n')
            for word, freq in freqTable.items():
            # print('\n' , word, freq)
               if word in sentence.lower(): 
                  # print('sentence_weight    ',sentence_weight)
                  if sentence in sentence_weight:     
                     sentence_weight[sentence] += freq 
                  else: 
                     sentence_weight[sentence] = freq 
                  #sentence_weight
               sumValues = 0
               for sentence in sentence_weight: 
                  sumValues += sentence_weight[sentence] 
               #sumValues 
               # Average value of a sentence from the original text 
               average = int(sumValues / len(sentence_weight)) 
               st.write(average,sumValues,len(sentence_weight),sep='\n\n')
               #average
               # Storing sentences into our summary. 
               summary = '' 
               counter=0
               for sentence in sentences: 
                  if (sentence in sentence_weight) and (sentence_weight[sentence] > (1.25* average)): 
                     summary += " " + sentence 
                     counter+=1
               st.write(counter,summary,sep='\n\n')
     
         
   uploaded_file = st.file_uploader("Choose a file",type=["csv"])
   if uploaded_file is not None:
      st.write("ORIGINAL CONTENT")
      st.write(type(uploaded_file))
      file_details = {"filename":uploaded_file.name,"filetype":uploaded_file.type,"filesize":uploaded_file.size}
      st.write(file_details)
      Df = pd.read_csv(uploaded_file)
      st.write("List Fiction Book")
      st.dataframe(Df)
      st.write("Shape")
      st.write(Df.shape)
      st.write("Info")
      buffer = io.StringIO()
      Df.info(buf=buffer)
      s = buffer.getvalue()
      st.text(s)
      
      if st.button('Summarize file'):
         st.info("Results")
         def remove_newlines_tabs(Df):
            # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
            Df['Description'] = Df['Description'].str.replace('\n\n\n\n', ' ')
            Df['Description'] = Df['Description'].str.replace('\n\n', ' ')
            Df['Description'] = Df['Description'].str.replace('\n', ' ')
            Df['Description'] = Df['Description'].str.replace('/', ' ')
            Df['Description'] = Df['Description'].str.replace('    ', ' ')
            Formatted_text = Df['Description'].str.replace('   ', ' ')
            return Formatted_text
         Df['Description'] = remove_newlines_tabs(Df)
         Df['Description'] = Df['Description'].replace('? ', '. ')
         Df['Description'] = Df['Description'].replace('*', '')
         Df['Description'] = Df['Description'].replace('\r', '')
         Df['Description'] = Df['Description'].replace('Page|', '')
         
         st.success('Replacing occurrences of tabs, line, special characters with a space.', icon="✅")
         if st.checkbox('Replacing occurrences of tabs, line, special characters with a space.'):
            st.write(Df['Description'])
            
         st.success('Word Tokenize', icon="✅")
         for i in range(len(Df)):
            sToken = nltk.word_tokenize(Df['Description'][i])
            st.write(i+1, "Description")
            st.write(sToken)
            
         stop = stopwords.words('english')
         Df['Description']= Df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
         st.success('Stopwords', icon="✅")
         st.write(sToken)
         st.write("List of stopwords:")
         stopwords = nltk.corpus.stopwords.words('english')
         st.write(stopwords[:10])
         
         st.success('Convert to lower case', icon="✅")
         Df['Description'] = Df['Description'].str.lower()
         st.write(sToken)
         
 
if choice == '📊 Result':
   st.info("Result (TXT file)")
   st.info("Result (CSV file)")
