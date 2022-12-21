import streamlit as st

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")

st.subheader("An extractive text summary system that generates summaries for a large number of texts")
st.title("ONLINE ENGLISH FICTION BOOK REVIEWS EXTRACTIVE TEXT SUMMARIZATION SYSTEM VIA MACHINE LEARNING APPROACHES")
st.write("he extractive text summarization system creates summaries by identifying, extracting the sentences, and subsequently combining the most important sentences in an online book to generate in form of a summary.")

with st.form("my_form"):
   st.write("Upload TXT File")
   title = st.text_input('Fiction Book Title')
   st.write('Fiction Books Content', title)
   st.form_submit_button("Summarize")

#custom funtion 
def summary(text):
    return summarize(text)

# Custom Components Fxn
def st_calculator(calc_html,width=1000,height=1350):
	calc_file = codecs.open(calc_html,'r')
	page = calc_file.read()
	components.html(page,width=width,height=height,scrolling=False)
