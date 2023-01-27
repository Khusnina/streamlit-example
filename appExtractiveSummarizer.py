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
import tensorflow as tf
import os
import seaborn as sns
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Flatten,Dropout,BatchNormalization,Embedding,Input,TimeDistributed
from tensorflow.keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
      def preprocess(text):
            text = text.lower() # lowercase
            text = text.split() # convert have'nt -> have not
            for i in range(len(text)):
               word = text[i]
               if word in contraction_mapping:
                  text[i] = contraction_mapping[word]
            text = " ".join(text)
            text = text.split()
            newtext = []
            for word in text:
               if word not in stop_words:
                  newtext.append(word)
            text = " ".join(newtext)
            text = text.replace("'s",'') # convert your's -> your
            text = re.sub(r'\(.*\)','',text) # remove (words)
            text = re.sub(r'[^a-zA-Z0-9. ]','',text) # remove punctuations
            text = re.sub(r'\.',' . ',text)
            return text
         
      Df['Description']=Df['Description'].apply(preprocess)
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
         st.text(df['Description'][8])
         content9 = df['Description'][8]
         st.download_button('Download', content9)
      book10 = st.checkbox('[2]Harry Potter - The Worst Birthday')
      if book10:
         st.write(df['Title'][9])
         st.text(df['Description'][9])
         content10 = df['Description'][9]
         st.download_button('Download', content10)
      book11 = st.checkbox('[3]Harry Potter - Owl Post')
      if book11:
         st.write(df['Title'][10])
         st.text(df['Description'][10])
         content11 = df['Description'][10]
         st.download_button('Download', content11)
      book12 = st.checkbox('[4]Harry Potter - The Riddle House')
      if book12:
         st.write(df['Title'][11])
         st.text(df['Description'][11])
         content12 = df['Description'][11]
         st.download_button('Download', content12)
      book13 = st.checkbox('[5]Harry Potter - Dudley Demented')
      if book13:
         st.write(df['Title'][12])
         st.text(df['Description'][12])
         content13 = df['Description'][12]
         st.download_button('Download', content13)
      book14 = st.checkbox('[6]Harry Potter - The Other Minister')
      if book14:
         st.write(df['Title'][13])
         st.text(df['Description'][13])
         content14 = df['Description'][13]
         st.download_button('Download', content14)
      book15 = st.checkbox('[7]Harry Potter - The Dark Lord Ascending')
      if book15:
         st.write(df['Title'][14])
         st.text(df['Description'][14])
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
         
         st.success('Word Tokenize', icon="✅")
         raw_text = raw_text.str.replace('\n\n\n\n', ' ')
         raw_text = raw_text.str.replace('\n\n', ' ')
         raw_text = raw_text.str.replace('\n', ' ')
         raw_text = raw_text.str.replace('/', ' ')
         raw_text = raw_text.str.replace('    ', ' ')
         raw_text = raw_text.str.replace('   ', ' ')
         raw_text = raw_text.replace('? ', '. ')
         raw_text = raw_text.replace('*', '')
         raw_text = raw_text.replace('\r', '')
         raw_text = raw_text.replace('Page|', '')
         
            
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
         #st.write("Items")
         #st.write(freqTable.items())
         #st.write(sorted(freqTable.items(), key = lambda x: x[1]))
         
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
               #st.write(average,sumValues,len(sentence_weight),sep='\n\n')
               #average
               # Storing sentences into our summary. 
               summary = '' 
               counter=0
               for sentence in sentences: 
                  if (sentence in sentence_weight) and (sentence_weight[sentence] > (1.25* average)): 
                     summary += " " + sentence 
                     counter+=1
               #st.write(counter,summary,sep='\n\n')
     
         
   uploaded_file = st.file_uploader("Choose a file",type=["csv"])
   if uploaded_file is not None:
      st.write("ORIGINAL CONTENT")
      type_file = type(uploaded_file)
      file_details = {"filename":uploaded_file.name,"filetype":uploaded_file.type,"filesize":uploaded_file.size}
      Df = pd.read_csv(uploaded_file)
      st.dataframe(Df)
      check_info = st.checkbox('File Information')
      if check_info:
         st.write("File Type")
         st.write(type_file)
         st.write("File Details")
         st.write(file_details)
         st.write("Shape")
         st.write(Df.shape)
         st.write("Info")
         buffer = io.StringIO()
         Df.info(buf=buffer)
         s = buffer.getvalue()
         st.text(s)
      
      if st.button('Summarize file'):
         st.info("Results")
         contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           
                            "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           
                            "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have", "chapter": "", "page" : ""}
         
         def clean_text(text):
            text=text.lower()
            text=' '.join([contraction_mapping[i] if i in contraction_mapping.keys() else i for i in text.split()])
            text=re.sub(r'\(.*\)',"",text)
            text=re.sub("'s","",text)
            text=re.sub('"','',text)
            text=' '.join([i for i in text.split() if i.isalpha()])
            text=re.sub('[^a-zA-Z]'," ",text)
            return text
         
         stop_words = stopwords.words('english')

         def preprocess(text):
            text = text.lower() # lowercase
            text = text.split() # convert have'nt -> have not
            for i in range(len(text)):
               word = text[i]
               if word in contraction_mapping:
                  text[i] = contraction_mapping[word]
            text = " ".join(text)
            text = text.split()
            newtext = []
            for word in text:
               if word not in stop_words:
                  newtext.append(word)
            text = " ".join(newtext)
            text = text.replace("'s",'') # convert your's -> your
            text = re.sub(r'\(.*\)','',text) # remove (words)
            text = re.sub(r'[^a-zA-Z0-9. ]','',text) # remove punctuations
            text = re.sub(r'\.',' . ',text)
            return text
         
         st.success('Cleaned', icon="✅")
         Df['Description']=Df['Description'].apply(clean_text)
         Df['Description']=Df['Description'].apply(preprocess)
         st.dataframe(Df)
         
         stop = stopwords.words('english')
         Df['Description']= Df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
         st.success('Stopwords', icon="✅")
         st.write("List of stopwords:")
         stopwords = nltk.corpus.stopwords.words('english')
         st.write(stopwords[:10])
         
         st.success('Word Tokenize', icon="✅")
         for i in range(len(Df)):
            sToken = nltk.word_tokenize(Df['Description'][i])
            st.write(i+1, "Description")
            st.write(sToken)
         
         #X_train,X_val,Y_train,Y_val=train_test_split(Df['Description'],Df['Title'],test_size=0.3,random_state=20)
         #st.write(len(X_train),len(Y_train))
         #st.write(len(X_val),len(Y_val))
         
         train_x, test_x, train_y, test_y = train_test_split(Df['Title'], Df['Description'], test_size=0.3, random_state=20)
         t_tokenizer = Tokenizer()
         t_tokenizer.fit_on_texts(list(train_x))

         thresh = 4
         count = 0
         total_count = 0
         frequency = 0
         total_frequency = 0

         for key, value in t_tokenizer.word_counts.items():
            total_count += 1
            total_frequency += value
            if value < thresh:
               count += 1
               frequency += value
         
         st.write("% of rare words in vocabulary: ", (count/total_count)*100.0)
         st.write("Total Coverage of rare words: ", (frequency/total_frequency)*100.0)
         s_max_features = total_count-count
         st.write("Summary Vocab: ", s_max_features)
         
         maxlen_text = 800
         maxlen_summ = 150
         val_x = test_x
         t_tokenizer = Tokenizer(num_words=t_max_features)
         t_tokenizer.fit_on_texts(list(train_x))
         train_x = t_tokenizer.texts_to_sequences(train_x)
         val_x = t_tokenizer.texts_to_sequences(val_x)

         train_x = pad_sequences(train_x, maxlen=maxlen_text, padding='post')
         val_x = pad_sequences(val_x, maxlen=maxlen_text, padding='post')
         
         val_y = test_y
         s_tokenizer = Tokenizer(num_words=s_max_features)
         s_tokenizer.fit_on_texts(list(train_y))
         train_y = s_tokenizer.texts_to_sequences(train_y)
         val_y = s_tokenizer.texts_to_sequences(val_y)

         train_y = pad_sequences(train_y, maxlen=maxlen_summ, padding='post')
         val_y = pad_sequences(val_y, maxlen=maxlen_summ, padding='post')
         
         st.write("Training Sequence", train_x.shape)
         st.write("Target Values Shape", train_y.shape)
         st.write("Test Sequence", val_x.shape)
         st.write("Target Test Shape", val_y.shape)
         
         def get_out_vector(text,summary,n=40):
            new_vec  = np.zeros(n)
            for txt in summary.split():
               if txt in text:
                  for i,word in enumerate(text.split()):
                     if word == txt:
                        new_vec[i] = 1
            return new_vec

         def get_summary(text,new_vec,thresh = 0.5):
            summary = []
            for i,word in enumerate(text.split()):
               if new_vec[i] >= thresh:
                  summary.append(word)
            return " ".join(summary)
         
         #vec = (get_out_vector(Df['Description'],Df['Title'],15))
         #st.write(get_summary(Df['Description'],vec))
         #book_len=[len(i.split()) for i in X_train]
         #sum_len=[len(i.split()) for i in Y_train]
         #plt.hist(book_len,bins=100)
         #plt.title('Book')
         #plt.show()
         #plt.hist(sum_len,bins=50)
         #plt.title('Summary')
         #plt.show()
         
if choice == '📊 Result':
   st.info("Result (TXT file)")
   st.info("Result (CSV file)")
