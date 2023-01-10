import streamlit as st  
import pandas as pd
import numpy as np
import io
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES</h1>", unsafe_allow_html=True)
st.markdown("<hr size='5' width='100%;'>", unsafe_allow_html=True)
activities = ["💡 Introduction","📚 Fiction Books","📝 Summarize","📊 Statistic"]
choice = st.sidebar.selectbox("Select Activity", activities)

if choice == '💡 Introduction':
   st.markdown("<h2 style='text-align: center; color: white;'>💡 INTRODUCTION 💡</h2>", unsafe_allow_html=True)
   st.markdown("<p style='text-align: center; color: white;'>Books are becoming increasingly popular among readers who love to read books especially English books. The passage of time has changed the source of books that can be read online in the form of electronic format where users only need to use a mobile device via the Internet. Books consist of various types of genres that are categorized into two namely fiction and non-fiction. Fiction books refer to literary plots, background, and characters designed and created from the writer’s imagination. Based on term of book sales in Malaysia, fiction books are more popular among readers than non-fiction book. Readers usually make an online book review after reading a whole book that contains a long story divided into several chapters that give structure and readability to the book by summarizing it manually to describe the contents.</p>", unsafe_allow_html=True)
   image = Image.open('summarization.png')
   col1, col2, col3 = st.columns([12,20,10])
   with col1:
      st.write("")
   with col2:
      st.image(image, caption='Text Summarization')
   with col3:
      st.write("")
   st.markdown("<p style='text-align: center; color: white;'>\nThe extractive text summarization system creates summaries by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary. The extraction summary method is the selection of sentences or phrases that have the highest score from the original text and organize them into a new, shorter text without changing the source text.</p>", unsafe_allow_html=True)
   st.markdown("<p style='text-align: center; color: white;'>\nFor extractive text summarization, the main sentence and the object are extracted without modifying the object itself, the strategy of concatenating on extracting summary from a given corpus. The system will select the most important sentences in the online book then combine them to form a summary extractively through machine learning with Natural Language Processing (NLP). The Repetitive Neural Network (RNN) is the model that will be used to generate the summary. The extracted summaries were used to train the extractive Recurrent Neural Network (RNN) model that had been used for the classification task and it generated the latest results. Recurrent Neural Network (RNN) is the most common architectures for Neural Text Summarization. Recurrent Neural Network (RNN) have been used widely in text summarization problem. The approach is based on the idea of identifying a set of sentences which collectively give the highest ROUGE with respect to the gold summary.</p>", unsafe_allow_html=True)
   
