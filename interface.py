import streamlit as st
from tkinter import *

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")

st.subheader("An extractive text summary system that generates summaries for a large number of texts")
st.title("ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES")
st.write("he extractive text summarization system creates summaries by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary.")

base = Tk()
box1= Entry(base)  
box1.place(x=200, y=120)  
