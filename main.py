import streamlit as st 
import pickle
import nltk
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def clean_code(text):
    text=re.sub(r'[^a-zA-Z0-9_\s]',' ',text) # Remove special characters
    tokens=word_tokenize(text.lower()) # Tokenization and lowercasing
    tokens=[word for word in tokens if word not in stopwords.words('english')] # Remove stopwords
    return ' '.join(tokens)
st.title("Bug Prediction and Prevention System")
st.write("Enter a python code snippet to check if it contains a bug.")
user_input=st.text_area("Enter code snippet : ")
button=st.button("Predict!")
if button:
    model=pickle.load(open("rf_model.pkl","rb"))
    vectorizer=pickle.load(open("vectorizer.pkl","rb"))
    cleaned_code=clean_code(user_input)
    input_tfidf=vectorizer.transform([cleaned_code])
    prediction=model.predict(input_tfidf)[0]
    if prediction:
        st.error("Bug Detected in the code")
    else:
        st.success("No Bugs Found in the code")



